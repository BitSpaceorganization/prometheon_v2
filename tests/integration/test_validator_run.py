"""`prometheon validator run`, end to end, with the weights worked out by hand.

Everything an operator touches is the shipped code path: the CLI entry point,
a TOML config built from ``configs/localnet.example.toml`` with its
``[scoring]`` section untouched, a policy file, a wallet on disk, the hub
client over a real socket to the fake hub, the epoch signature, the gzip
files, the labelling client over a real socket, the registry checks against
Hugging Face's API shape, scoring, the state file, ``set_weights`` and the
signed record.

Three stand-ins, each at the edge of the process: the chain is an object with
the SDK's accessors that records ``set_weights``; Hugging Face is its API shape
behind ``httpx.MockTransport``; and the model engine is a double that gets a
fixed set of items wrong, because CI does not install torch. The local-chain
run in ``tools/localnet/cycle.py`` replaces all three with the real thing.

Every expected number below is derived in the comments from spec section 2:
70% to data in proportion to ``2 * min(V_pos, V_neg) * (V_pos + V_neg) / S``,
25% / 4% / 1% to the first three models, largest remainder throughout.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import httpx
import pytest
from bittensor_wallet import Keypair, Wallet

from prometheon.evaluation.engine import ModerationResult, Verdict
from prometheon.hub.auth import CallerRole, verify_evaluation
from prometheon.hub.epoch import epoch_content_hash
from prometheon.hub.fake import FakeHub, serve
from prometheon.hub.models import EvaluationSubmission
from prometheon.hub.normalize import normalized_sha256
from prometheon.registry.huggingface import HuggingFaceClient

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
NETUID = 7
DAY = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
POLICY_TEXT = "# Policy\n\nVersion: e2e-1\n\nAnswer YES if the item threatens or abuses.\n"

OWNER = Keypair.create_from_uri("//E2E-Owner")
VALIDATOR_URI = "//E2E-Validator"
VALIDATOR = Keypair.create_from_uri(VALIDATOR_URI)
MINERS = {
    name: (Keypair.create_from_uri(f"//E2E-{name}"), Keypair.create_from_uri(f"//E2E-cold-{name}"))
    for name in ("A", "B", "C", "D")
}
#: uid order on the chain: the owner, the validator, then A, B, C, D.
UIDS = {"A": 2, "B": 3, "C": 4, "D": 5}
#: (verified active members, members). D is one member short of 30.
GROUPS = {"A": (14, 40), "B": (12, 31), "C": (10, 30), "D": (14, 29)}

_WORDS = (
    "harbour lantern copper meadow violet engine signal orchard basalt cedar "
    "falcon granite hollow juniper kestrel lagoon marble nickel oyster pepper "
    "quartz raven saffron thistle umber velvet willow yonder zephyr amber "
    "beacon cobalt dune ember fjord glacier heron ivory jasper kelp"
).split()


def _body(seed: str, violating: bool) -> str:
    """Distinct text per seed: far apart in SimHash, so nothing is a near-duplicate."""
    digest = hashlib.sha256(seed.encode()).digest()
    words = " ".join(_WORDS[byte % len(_WORDS)] for byte in digest[:14])
    return f"{'I will hurt you' if violating else 'Lovely day'}: {words} ({seed})"


# --- the edges: chain, Hugging Face, model engine, labeller -----------------


class Chain:
    """The SDK accessors the validator drives, recording what it sends."""

    def __init__(self) -> None:
        self.hotkeys = [
            OWNER.ss58_address,
            VALIDATOR.ss58_address,
            *(MINERS[name][0].ss58_address for name in "ABCD"),
        ]
        self.sent: list[dict[str, Any]] = []

    def metagraph(self, netuid: int, lite: bool = True) -> Any:
        return SimpleNamespace(
            neurons=[],
            hotkeys=list(self.hotkeys),
            uids=list(range(len(self.hotkeys))),
            validator_permit=[False, True] + [False] * (len(self.hotkeys) - 2),
            stake=[0.0, 1000.0] + [1.0] * (len(self.hotkeys) - 2),
            block=5_000,
        )

    def get_subnet_hyperparameters(self, netuid: int) -> Any:
        return SimpleNamespace(
            weights_version=0,
            commit_reveal_weights_enabled=False,
            min_allowed_weights=1,
            max_weight_limit=65_535,
        )

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return OWNER.ss58_address

    def get_commitment(self, netuid: int, uid: int) -> str:
        return ""

    def weights_rate_limit(self, netuid: int) -> int:
        return 0

    def set_weights(
        self,
        *,
        wallet: Any,
        netuid: int,
        uids: list[int],
        weights: list[int],
        version_key: int,
        wait_for_inclusion: bool = True,
        mechid: int = 0,
    ) -> Any:
        self.sent.append({"netuid": netuid, "uids": uids, "weights": weights, "mechid": mechid})
        return SimpleNamespace(
            success=True, message="Finalized.", error=None, extrinsic_receipt=None
        )


def _lfs(repo: str, shard: int) -> str:
    return hashlib.sha256(f"{repo}:{shard}".encode()).hexdigest()


def _model(name: str) -> dict[str, Any]:
    """A two-shard model; its fingerprint by the spec's formula, not the code's."""
    repo = f"org-{name.lower()}/guard"
    lfs = sorted(_lfs(repo, shard) for shard in (1, 2))
    return {
        "hotkey": MINERS[name][0].ss58_address,
        "repo": repo,
        "revision": hashlib.sha1(repo.encode()).hexdigest(),  # noqa: S324 - a fixture id
        "commit_block": 100 * UIDS[name],
        "fingerprint": hashlib.sha256("\n".join(lfs).encode()).hexdigest(),
        "payment": "paid",
    }


def _huggingface(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.startswith("/api/models/") and "/revision/" in path:
        repo, revision = path.removeprefix("/api/models/").split("/revision/")
        siblings: list[dict[str, Any]] = [
            {"rfilename": "config.json", "size": 700},
            {"rfilename": "tokenizer.json", "size": 9_000},
            {"rfilename": "model.safetensors.index.json", "size": 2_000},
        ]
        for shard in (1, 2):
            sha = _lfs(repo, shard)
            size = 400_000_000
            siblings.append(
                {
                    "rfilename": f"model-0000{shard}-of-00002.safetensors",
                    "size": size,
                    "lfs": {"sha256": sha, "size": size, "pointerSize": 134},
                }
            )
        return httpx.Response(200, json={"id": repo, "sha": revision, "siblings": siblings})
    if path.endswith("/config.json"):
        return httpx.Response(200, json={"model_type": "qwen2"})
    return httpx.Response(404, json={"error": "not found"})


class Engine:
    """Answers from the truth table, except on the items it is set to miss."""

    def __init__(self, truth: dict[str, bool], misses: frozenset[str]) -> None:
        self.truth = truth
        self.misses = misses

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def moderate(self, policy: str, items: Any) -> ModerationResult:
        verdicts = tuple(
            Verdict(item_id=item_id, violates=self.truth[item_id] != (item_id in self.misses))
            for item_id, _content in items
        )
        return ModerationResult(
            verdicts=verdicts, prompt_tokens=100 * len(verdicts), completion_tokens=len(verdicts)
        )


def _labeller(truth: dict[str, bool]) -> type[BaseHTTPRequestHandler]:
    class Labeller(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            user = body["messages"][-1]["content"]
            # One post per request, on the line after the opening fence.
            (item,) = [json.loads(line) for line in user.splitlines() if line.startswith('{"id"')]
            content = json.dumps({"id": item["id"], "violates": truth[item["id"]]})
            payload = json.dumps(
                {
                    "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    return Labeller


# --- the config file --------------------------------------------------------


def _toml(data: dict[str, Any]) -> str:
    def value(item: Any) -> str:
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, int | float):
            return repr(item)
        if isinstance(item, str):
            return json.dumps(item)
        return "[" + ", ".join(value(element) for element in item) + "]"

    lines: list[str] = []

    def table(name: str, body: dict[str, Any]) -> None:
        lines.append(f"[{name}]")
        lines.extend(
            f"{json.dumps(key)} = {value(item)}"
            for key, item in body.items()
            if not isinstance(item, dict)
        )
        for key, item in body.items():
            if isinstance(item, dict):
                table(f"{name}.{key}", item)

    for name, body in data.items():
        table(name, body)
    return "\n".join(lines) + "\n"


# --- the day ----------------------------------------------------------------


#: Per miner: (claimed violating, labelled violating) for each of its items.
Claims = dict[str, list[tuple[bool, bool]]]

#: A: 6 violating and 4 compliant, all confirmed.
#: B: 5 and 5, all confirmed.
#: C: 3 violating confirmed; 7 claimed compliant, of which the labeller finds
#:    2 violating, so 5 compliant confirmed.
#: D: 5 and 5, all confirmed, but one member short of eligible.
STANDARD: Claims = {
    "A": [(True, True)] * 6 + [(False, False)] * 4,
    "B": [(True, True)] * 5 + [(False, False)] * 5,
    "C": [(True, True)] * 3 + [(False, False)] * 5 + [(False, True)] * 2,
    "D": [(True, True)] * 5 + [(False, False)] * 5,
}

#: Every eligible miner claims only violating items: min(V_pos, V_neg) = 0.
ONE_SIDED: Claims = {
    "A": [(True, True)] * 10,
    "B": [(True, True)] * 10,
    "C": [(True, True)] * 10,
    "D": [(True, True)] * 5 + [(False, False)] * 5,
}

#: Production items each model gets wrong. Every model sees all of them, so
#: the order is A, then C, then B, whatever else is in each corpus.
MISSES = {
    "A": frozenset(),
    "B": frozenset(f"p{index:02d}" for index in range(6)),
    "C": frozenset(f"p{index:02d}" for index in range(3)),
    "D": frozenset(),
}


@pytest.fixture
def run_day(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    servers: list[ThreadingHTTPServer] = []

    def run(claims: Claims) -> dict[str, Any]:
        truth: dict[str, bool] = {}
        tests: list[dict[str, Any]] = []
        for name, items in claims.items():
            hotkey = MINERS[name][0].ss58_address
            for index, (claimed, labelled) in enumerate(items):
                item_id = f"t{name}{index:02d}"
                body = _body(item_id, labelled)
                truth[item_id] = labelled
                tests.append(
                    {
                        "id": item_id,
                        "miner_hotkey": hotkey,
                        "author": f"anon-{item_id}",
                        "claimed_violating": claimed,
                        "body": body,
                        "normalized_sha256": normalized_sha256(body),
                        "created_at": f"{DAY.isoformat()}T{8 + index:02d}:{UIDS[name]:02d}:00Z",
                        "verified": True,
                    }
                )
        # 80 production items, 1 in 8 violating: with 40 test items, every
        # model is scored on at least 110, above the shipped floor of 100.
        productions = []
        for index in range(80):
            item_id = f"p{index:02d}"
            truth[item_id] = index % 8 == 7
            productions.append(
                {
                    "id": item_id,
                    "body": _body(item_id, truth[item_id]),
                    "created_at": f"{DAY.isoformat()}T06:{index % 60:02d}:00Z",
                }
            )

        hub = FakeHub(netuid=NETUID)
        hub.register(VALIDATOR.ss58_address, CallerRole.VALIDATOR)
        miners = [
            {
                "hotkey": hotkey.ss58_address,
                "uid": uid,
                "coldkey": hotkey.ss58_address,
                "verified_active_members": 0,
                "members": 0,
                "eligible": False,
            }
            for uid, hotkey in enumerate((OWNER, VALIDATOR))
        ]
        for name in "ABCD":
            verified, members = GROUPS[name]
            miners.append(
                {
                    "hotkey": MINERS[name][0].ss58_address,
                    "uid": UIDS[name],
                    "coldkey": MINERS[name][1].ss58_address,
                    "verified_active_members": verified,
                    "members": members,
                    "eligible": verified >= 10 and members >= 30,
                }
            )
        raw_epoch = hub.publish_day(
            DAY,
            miners=miners,
            models=[_model(name) for name in "ABCD"],
            test_lines=tests,
            production_lines=productions,
            policy_version="e2e-1",
            policy_sha256=hashlib.sha256(POLICY_TEXT.encode()).hexdigest(),
            frozen_block=4_990,
        )
        hub_server = serve(hub, port=0)
        labeller = ThreadingHTTPServer(("127.0.0.1", 0), _labeller(truth))
        threading.Thread(target=labeller.serve_forever, daemon=True).start()
        servers.extend((hub_server, labeller))

        # A wallet on disk, where the CLI looks for it.
        monkeypatch.setenv("HOME", str(tmp_path))
        wallet = Wallet(
            name="validator", hotkey="default", path=str(tmp_path / ".bittensor" / "wallets")
        )
        wallet.create_coldkey_from_uri(
            "//E2E-ValidatorCold", use_password=False, overwrite=True, suppress=True
        )
        wallet.create_hotkey_from_uri(
            VALIDATOR_URI, use_password=False, overwrite=True, suppress=True
        )

        example = tomllib.loads((ROOT / "configs" / "localnet.example.toml").read_text())
        example["chain"]["netuid"] = NETUID
        example["hub"].update(
            base_url=f"http://127.0.0.1:{hub_server.server_address[1]}",
            epoch_keys=hub.epoch_keys,
        )
        example["labelling"].update(
            model="e2e-labeller",
            base_url=f"http://127.0.0.1:{labeller.server_address[1]}/v1",
            api_key_env="E2E_LABELLER_KEY",
            temperature=0.0,
        )
        example["validator"].update(
            state_directory=str(tmp_path / "state"),
            submit_weights=True,
            dry_run=False,
            submit_results=True,
        )
        config = tmp_path / "validator.toml"
        config.write_text(_toml(example))
        policy = tmp_path / "policy.md"
        policy.write_text(POLICY_TEXT)
        monkeypatch.setenv("E2E_LABELLER_KEY", "local")

        chain = Chain()
        monkeypatch.setattr(
            "prometheon.chain.subtensor._bittensor",
            lambda: SimpleNamespace(Subtensor=lambda network: chain),
        )
        monkeypatch.setattr(
            "prometheon.cli.validator.HuggingFaceClient",
            lambda: HuggingFaceClient(
                transport=httpx.MockTransport(_huggingface), sleep=lambda _s: None
            ),
        )
        by_path = {f"/models/{MINERS[name][0].ss58_address}": name for name in "ABCD"}
        monkeypatch.setattr(
            "prometheon.cli.cycle.download_checkpoint",
            lambda target, **_k: f"/models/{target.hotkey}",
        )
        monkeypatch.setattr(
            "prometheon.cli.cycle.build_engine",
            lambda path, **_k: Engine(truth, MISSES[by_path[path]]),
        )
        monkeypatch.setattr("prometheon.cli.cycle.check_evaluation_runtime", lambda **_k: None)

        cli_main = importlib.import_module("prometheon.cli.main")
        argv = ["validator", "run", "--config", str(config), "--date", DAY.isoformat()]
        code = cli_main.main([*argv, "--policy", str(policy)])

        state = json.loads((tmp_path / "state" / "last-weights.json").read_text())
        stored = hub.results.get((DAY, VALIDATOR.ss58_address))
        return {
            "code": code,
            "sent": chain.sent,
            "state": state,
            "raw_record": stored,
            "raw_epoch": raw_epoch,
        }

    yield run
    for server in servers:
        server.shutdown()


def _hotkey(name: str) -> str:
    return MINERS[name][0].ss58_address


def _record(day: dict[str, Any]) -> EvaluationSubmission:
    assert day["raw_record"] is not None, "no record reached the hub"
    record = EvaluationSubmission.model_validate_json(day["raw_record"])
    verify_evaluation(record)
    assert record.epoch_content_hash == epoch_content_hash(day["raw_epoch"])
    return record


def test_a_standard_day_on_chain_matches_the_spec_by_hand(run_day: Any) -> None:
    day = run_day(STANDARD)
    assert day["code"] == 0

    # Data, 70% = 700,000 units, D excluded (29 members):
    #   A: amount 2 * min(6, 4) = 8,  quality 10/10 -> score 8
    #   B: amount 2 * min(5, 5) = 10, quality 10/10 -> score 10
    #   C: amount 2 * min(3, 5) = 6,  quality 8/10  -> score 4.8
    #   total 22.8:  A 245,614.04  B 307,017.54  C 147,368.42
    #   floors sum to 699,999; the one left over goes to the largest
    #   remainder, B: A 245,614, B 307,018, C 147,368.
    # Models: A misses nothing, C misses 3, B misses 6, D is not evaluated.
    #   A first 250,000; C second 40,000; B third 10,000. Nothing burns.
    expected = {
        _hotkey("A"): 245_614 + 250_000,
        _hotkey("B"): 307_018 + 10_000,
        _hotkey("C"): 147_368 + 40_000,
    }
    assert day["state"]["sent_weights"] == expected

    # u16, x 65535 / 1,000,000:
    #   A 32,480.063  B 20,775.775  C 12,279.162 -> floors 65,534, B +1.
    assert day["sent"] == [
        {"netuid": NETUID, "uids": [2, 3, 4], "weights": [32_480, 20_776, 12_279], "mechid": 0}
    ]

    record = _record(day)
    assert record.burned_weight == 0
    rows = {row.hotkey: row for row in record.results}
    assert {hotkey: row.weight for hotkey, row in rows.items() if row.weight} == expected
    assert [
        (rows[_hotkey(n)].dataset_violating_confirmed, rows[_hotkey(n)].dataset_compliant_confirmed)
        for n in "ABC"
    ] == [(6, 4), (5, 5), (3, 5)]
    assert [rows[_hotkey(n)].dataset_score_micro for n in "ABC"] == [
        8_000_000,
        10_000_000,
        4_800_000,
    ]
    assert [rows[_hotkey(n)].model_rank for n in "ACB"] == [1, 2, 3]
    assert rows[_hotkey("D")].model_status.value == "ineligible"
    assert rows[_hotkey("D")].weight == 0


def test_a_day_with_no_data_winner_burns_exactly_the_data_pool(run_day: Any) -> None:
    day = run_day(ONE_SIDED)
    assert day["code"] == 0

    # Every eligible miner has V_neg = 0, so every data score is 0 and the
    # 70% burns to the owner hotkey. The model pool pays as before.
    expected = {
        OWNER.ss58_address: 700_000,
        _hotkey("A"): 250_000,
        _hotkey("C"): 40_000,
        _hotkey("B"): 10_000,
    }
    assert day["state"]["sent_weights"] == expected

    # u16: owner 45,874.5  A 16,383.75  B 655.35  C 2,621.4 -> floors 65,533;
    # the two left over go to the largest remainders, A (.75) then owner (.5).
    assert day["sent"] == [
        {
            "netuid": NETUID,
            "uids": [0, 2, 3, 4],
            "weights": [45_875, 16_384, 655, 2_621],
            "mechid": 0,
        }
    ]

    record = _record(day)
    assert record.burned_weight == 700_000
    rows = {row.hotkey: row.weight for row in record.results if row.weight}
    assert {**rows, OWNER.ss58_address: record.burned_weight} == expected
