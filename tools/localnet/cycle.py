"""A whole Prometheon cycle on a local subtensor chain, with real extrinsics.

What this proves, end to end, on a chain nobody else uses:

1. a subnet is registered and two miners join it;
2. each miner buys a submission credit with ``prometheon credits buy`` (a real
   ``transfer_stake`` to a local treasury coldkey) and commits a real Hugging
   Face model with ``prometheon model commit``;
3. a local hub (the in-repo fake, served over HTTP) freezes the day into a
   signed epoch from what the chain now says, with content files built here;
4. ``prometheon validator run`` fetches and verifies the epoch, labels the day
   against a local labeller that answers from a fixed truth table, downloads
   and runs both models on this machine's CPU, scores, sets weights on chain
   and publishes its signed record to the hub;
5. the weights are read back from the chain and checked three ways: against
   the vector the validator saved before sending, against the record it
   published, and against the spec's split computed by hand from the record.

Requirements: the opentensor subtensor localnet image published on a local
port (see docs/localnet.md), network access to Hugging Face, and a Python
environment with this package and the ``wrapper`` extra (torch, transformers).

    python tools/localnet/cycle.py --endpoint ws://127.0.0.1:9945 --workdir /tmp/run

``--min-allowed-weights 3`` makes the chain demand more weights than the day
earns, so the run also proves the vector is shaped to the subnet's limits and
that the chain accepts it (S5).

Each run registers a new subnet, so it can be repeated on the same chain.
Nothing here touches a public network: //Alice, //Bob and the other dev
accounts exist only on the local chain, and their keys are public by design.
"""

from __future__ import annotations

import argparse
import datetime as dt
import decimal
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import bittensor
from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.utils.balance import Balance
from bittensor.utils.weight_utils import convert_and_normalize_weights_and_uids
from bittensor_wallet import Wallet

from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import resolve_allocations, to_u16_chain_vector
from prometheon.cli.validator import _policy_text
from prometheon.hub.auth import CallerRole
from prometheon.hub.fake import FakeHub, serve
from prometheon.hub.normalize import normalized_sha256
from prometheon.registry.huggingface import HuggingFaceClient, weights_fingerprint
from prometheon.version import SUBMISSION_FEE_ALPHA

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "localnet.example.toml"
TOTAL_UNITS = 1_000_000

#: The day holds a few dozen items, so the shipped floor of 100 would rank no
#: model. The run lowers it to this, and changes nothing else in [scoring].
MIN_SCORED_ITEMS = 10

#: Two models that pass the manifest: weights in LFS, a config, a tokenizer,
#: no code. Small enough to run on a CPU.
MODELS = {
    "miner-a": ("HuggingFaceTB/SmolLM2-135M", "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"),
    "miner-b": ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
}

#: Group sizes the epoch reports: (verified active members, members). Both
#: clear the floors of 10 and 30, so both miners are eligible.
GROUPS = {"miner-a": (14, 40), "miner-b": (12, 31)}

VIOLATING = [
    "you are a worthless idiot and everyone here should ignore you",
    "send me your wallet seed phrase or I will leak your address",
    "people from that country are vermin and should be thrown out",
    "I know where you live and I am coming for you tonight",
    "buy followers now cheap cheap cheap click this link http://spam.example",
    "shut up you pathetic loser nobody wants you on this forum",
    "this whole group is subhuman trash and deserves to suffer",
    "drop your private key here and I will double your tokens",
    "I will find your family and make them pay for your post",
    "free tokens for everyone who clicks http://phish.example right now",
]
COMPLIANT = [
    "the validator update shipped yesterday and the logs look clean",
    "does anyone have a good guide for staking on a new subnet",
    "I disagree with the proposal but I respect the reasoning behind it",
    "great write-up on the scoring rules, the examples helped a lot",
    "my node restarted overnight, any idea why the peers dropped",
    "the meetup next week is in the main hall at six in the evening",
    "thanks for answering my question about the emission split",
    "I think the dashboard should show the burn line more clearly",
    "the new model is slower on my laptop but the answers are better",
    "welcome to the group, the pinned post explains how things work",
]
PRODUCTION = [
    ("good morning everyone, coffee first and then the release notes", False),
    ("the price chart looks flat today, nothing to report", False),
    ("has anyone tried the new wallet extension on firefox", False),
    ("reading the subnet docs tonight, they are clearer than before", False),
    ("the team call moved to thursday this week", False),
    ("lovely sunset over the harbour this evening", False),
    ("I finally got my validator synced after two days", False),
    ("remember to back up your keys before upgrading", False),
    ("the conference talk about moderation was really interesting", False),
    ("small reminder that the forum rules are pinned at the top", False),
    ("you people are disgusting animals and should all die", True),
    ("the new update fixed the memory leak I reported", False),
]


def log(message: str) -> None:
    print(f"[localnet] {message}", flush=True)


def check(response: Any, what: str) -> None:
    if not getattr(response, "success", False):
        raise SystemExit(f"{what} failed: {getattr(response, 'message', response)}")


def landed(response: Any, what: str, effect: Callable[[], bool]) -> None:
    """Accept a write whose effect is on chain, whatever the SDK reported.

    The localnet image runs two block authors with fast blocks, so short forks
    are routine. The SDK reads state again after a write lands, at a block
    hash taken before it, and that read sometimes fails with "State already
    discarded" although the extrinsic is in. So the effect itself is checked.
    """
    if getattr(response, "success", False):
        return
    if effect():
        log(f"{what}: on chain, although the SDK reported {getattr(response, 'message', '')!r}")
        return
    raise SystemExit(f"{what} failed: {getattr(response, 'message', response)}")


def stake_on(subtensor: Any, wallet: Wallet, netuid: int) -> int:
    return int(
        subtensor.get_stake(
            coldkey_ss58=wallet.coldkeypub.ss58_address,
            hotkey_ss58=wallet.hotkey.ss58_address,
            netuid=netuid,
        ).rao
    )


def add_stake(subtensor: Any, wallet: Wallet, netuid: int, tao: int, what: str) -> None:
    before = stake_on(subtensor, wallet, netuid)
    response = subtensor.add_stake(
        wallet, netuid, wallet.hotkey.ss58_address, Balance.from_tao(tao)
    )
    landed(response, what, lambda: stake_on(subtensor, wallet, netuid) > before)


def dev_wallet(home: Path, name: str, coldkey_uri: str, hotkey_uri: str | None) -> Wallet:
    """A wallet under ``home``, from a dev URI: public test keys, local chain only."""
    created = Wallet(name=name, hotkey="default", path=str(home / ".bittensor" / "wallets"))
    created.create_coldkey_from_uri(coldkey_uri, use_password=False, overwrite=True)
    if hotkey_uri is not None:
        created.create_hotkey_from_uri(hotkey_uri, use_password=False, overwrite=True)
    return created


def wait_blocks(subtensor: Any, count: int) -> None:
    target = subtensor.get_current_block() + count
    while subtensor.get_current_block() < target:
        time.sleep(0.25)


def view_of(subtensor: Any, netuid: int) -> MetagraphView:
    return MetagraphView.from_sdk_metagraph(
        subtensor.metagraph(netuid=netuid, lite=False), netuid=netuid
    )


# --- local services ---------------------------------------------------------


class Labeller(BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that labels from a fixed truth table."""

    truth: dict[str, bool] = {}  # noqa: RUF012 - set once before serving

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        user = body["messages"][-1]["content"]
        # One post per request, on the line after the opening fence. The
        # request carries its id and text only, never the author's claim.
        (item,) = [json.loads(line) for line in user.splitlines() if line.startswith('{"id"')]
        assert set(item) == {"id", "content"}, f"the labeller was sent {sorted(item)}"
        answer = {"id": item["id"], "violates": self.truth[item["id"]]}
        payload = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(answer)},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        return None


# --- config -----------------------------------------------------------------


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"no TOML form for {value!r}")


def to_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def table(name: str, body: dict[str, Any]) -> None:
        lines.append(f"[{name}]")
        lines.extend(
            f"{json.dumps(key)} = {_toml_value(value)}"
            for key, value in body.items()
            if not isinstance(value, dict)
        )
        lines.append("")
        for key, value in body.items():
            if isinstance(value, dict):
                table(f"{name}.{key}", value)

    for name, body in data.items():
        table(name, body)
    return "\n".join(lines)


def write_config(path: Path, **sections: dict[str, Any]) -> Path:
    """The shipped localnet example with this run's values filled in.

    Starting from the example rather than a private template means this run
    also proves the example loads and runs once its placeholders are filled.
    """
    config = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    path.write_text(to_toml(config), encoding="utf-8")
    return path


def cli(home: Path, *args: str) -> None:
    """Run the real ``prometheon`` command, with wallets under ``home``."""
    env = {
        **os.environ,
        "HOME": str(home),
        "LOCAL_LABELLER_KEY": "local",
        # Reuse the operator's model cache rather than downloading again.
        "HF_HOME": os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")),
    }
    # The installed console script, exactly as an operator runs it.
    script = Path(sys.executable).with_name("prometheon")
    command = [str(script)] if script.exists() else [sys.executable, "-m", "prometheon.cli.main"]
    result = subprocess.run(  # noqa: S603 - a fixed argv, no shell
        [*command, *args],
        env=env,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    for line in (result.stdout + result.stderr).splitlines():
        print(f"    | {line}", flush=True)
    if result.returncode != 0:
        raise SystemExit(f"prometheon {' '.join(args[:2])} exited {result.returncode}")


# --- the spec, by hand --------------------------------------------------------


def penalty_by_hand(parameters: int) -> int:
    """floor(100 * log2(P / 500M)), 0 up to 500M, in 60-digit decimals."""
    if parameters <= 500_000_000:
        return 0
    with decimal.localcontext() as context:
        context.prec = 60
        ratio = decimal.Decimal(parameters) / decimal.Decimal(500_000_000)
        exact = 100 * ratio.ln() / decimal.Decimal(2).ln()
        return int(exact.to_integral_value(rounding=decimal.ROUND_FLOOR))


def ranked_by_hand(record: dict[str, Any], commit_blocks: dict[str, int]) -> list[str]:
    """docs/scoring.md "Ranks and places", applied to the record's own numbers.

    Each score is checked as accuracy less the size penalty, recomputed here.
    Then each place goes to the earliest commit, then the lower hotkey, among
    the models within 100 bp of the best remaining score.
    """
    field: dict[str, int] = {}
    for row in record["results"]:
        if row["model_status"] != "evaluated":
            continue
        penalty = penalty_by_hand(row["model_parameters"])
        assert row["size_penalty_bp"] == penalty, f"{row['hotkey']}: penalty {row}"
        assert row["model_score_bp"] == max(0, row["raw_accuracy_bp"] - penalty), row
        if row["model_score_bp"] > 0 and row["model_items_scored"] >= MIN_SCORED_ITEMS:
            field[row["hotkey"]] = row["model_score_bp"]
    places: list[str] = []
    while field:
        best = max(field.values())
        winner = min(
            (hotkey for hotkey, score in field.items() if score >= best - 100),
            key=lambda hotkey: (commit_blocks[hotkey], hotkey),
        )
        places.append(winner)
        del field[winner]
    return places


def expected_units(
    record: dict[str, Any], commit_blocks: dict[str, int]
) -> tuple[dict[str, int], int]:
    """Spec section 2 applied to the record's published inputs.

    Both miners submit 5 items each way, every claim true, nothing dropped, so
    each data score is 2 * min(5, 5) * 10/10 = 10 and the 70% splits evenly.
    Models: placed by hand, then 25% / 4% / 1% of the total, an unfilled place
    going to the lowest-placed winner present. Whatever is left burns.
    """
    rows = [row for row in record["results"] if row["dataset_submitted_count"]]
    for row in rows:
        counts = (
            row["dataset_submitted_count"],
            row["dataset_violating_confirmed"],
            row["dataset_compliant_confirmed"],
        )
        assert counts == (10, 5, 5), f"unexpected data counts for {row['hotkey']}: {counts}"
    units = {row["hotkey"]: 700_000 // len(rows) for row in rows}
    places = ranked_by_hand(record, commit_blocks)
    published = {row["hotkey"]: row["model_rank"] for row in record["results"]}
    assert all(published[hotkey] == place for place, hotkey in enumerate(places, start=1)), (
        f"the record places the models {published}, by hand {places}"
    )
    ranked = [row for hotkey in places for row in record["results"] if row["hotkey"] == hotkey]
    shares = [250_000, 40_000, 10_000][: len(ranked)]
    if shares:
        shares[-1] += 300_000 - sum(shares)
    for row, share in zip(ranked, shares, strict=True):
        units[row["hotkey"]] = units.get(row["hotkey"], 0) + share
    return units, TOTAL_UNITS - sum(units.values())


def padded_by_hand(
    units: dict[str, int], *, view: MetagraphView, validator: str, min_allowed_weights: int
) -> dict[str, int]:
    """docs/scoring.md "The subnet's weight limits", applied independently of the code.

    Too few uids: the lowest-numbered registered uids not already in the vector,
    the validator's own only if nothing else is left, each get
    ceil(1,000,000 / 65535) = 16 units, taken from the largest entry.
    """
    required = min(len(view), min_allowed_weights)
    present = {hotkey: amount for hotkey, amount in units.items() if amount > 0}
    missing = required - len(present)
    if missing <= 0:
        return present
    order = [hotkey for _uid, hotkey in sorted(zip(view.uids, view.hotkeys, strict=True))]
    candidates = [hotkey for hotkey in order if hotkey not in present and hotkey != validator]
    if len(candidates) < missing:
        candidates.append(validator)
    pad = -(-TOTAL_UNITS // 65_535)
    donor = max(present, key=lambda hotkey: (present[hotkey], hotkey))
    for hotkey in candidates[:missing]:
        present[hotkey] = present.get(hotkey, 0) + pad
    present[donor] -= pad * missing
    return present


# --- the run ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--endpoint", default="ws://127.0.0.1:9945")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--hub-port", type=int, default=8600)
    parser.add_argument("--labeller-port", type=int, default=8601)
    parser.add_argument(
        "--min-allowed-weights",
        type=int,
        default=1,
        help="set the subnet's min_allowed_weights, to exercise vector shaping (S5)",
    )
    args = parser.parse_args()

    home = args.workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    subtensor = bittensor.Subtensor(network=args.endpoint)
    log(f"chain {args.endpoint} at block {subtensor.get_current_block()}")

    # --- 1. a subnet, its owner (who also validates) and two miners -------
    owner = dev_wallet(home, "owner", "//Alice", "//Alice//prometheon-validator")
    miners = {
        "miner-a": dev_wallet(home, "miner-a", "//Bob", "//Bob//prometheon-miner"),
        "miner-b": dev_wallet(home, "miner-b", "//Charlie", "//Charlie//prometheon-miner"),
    }
    treasury = dev_wallet(home, "treasury", "//Eve", None).coldkeypub.ss58_address

    before = set(subtensor.get_all_subnets_netuid())
    check(subtensor.register_subnet(owner), "register_subnet")
    (netuid,) = set(subtensor.get_all_subnets_netuid()) - before
    log(f"registered netuid {netuid}; owner and validator hotkey {owner.hotkey.ss58_address}")
    for attempt in range(60):
        if subtensor.start_call(owner, netuid).success:
            break
        if attempt == 59:
            raise SystemExit("start_call never succeeded")
        wait_blocks(subtensor, 5)
    # The localnet image's tempo is shorter than its admin freeze window, so no
    # hyperparameter can change until the window is lifted. It is chain-wide
    # and the chain may be shared, so it is put back straight afterwards.
    freeze = subtensor.substrate.query("SubtensorModule", "AdminFreezeWindow").value
    for function, params in (
        ("sudo_set_admin_freeze_window", {"window": 0}),
        ("sudo_set_commit_reveal_weights_enabled", {"netuid": netuid, "enabled": False}),
        ("sudo_set_weights_set_rate_limit", {"netuid": netuid, "weights_set_rate_limit": 0}),
        ("sudo_set_tempo", {"netuid": netuid, "tempo": 20}),
        (
            "sudo_set_min_allowed_weights",
            {"netuid": netuid, "min_allowed_weights": args.min_allowed_weights},
        ),
        ("sudo_set_admin_freeze_window", {"window": freeze}),
    ):
        check(sudo_call_extrinsic(subtensor, owner, function, params), function)
    # A sudo extrinsic succeeds even when the call inside it fails, so read back.
    assert not subtensor.commit_reveal_enabled(netuid=netuid), "commit-reveal is still on"
    assert subtensor.weights_rate_limit(netuid=netuid) == 0, "the rate limit did not change"
    assert subtensor.tempo(netuid=netuid) == 20, "the tempo did not change"
    limits = subtensor.get_subnet_hyperparameters(netuid=netuid)
    assert limits.min_allowed_weights == args.min_allowed_weights, "min_allowed_weights unchanged"
    log(
        "subnet started; commit-reveal off, no weights rate limit, tempo 20, "
        f"min_allowed_weights {limits.min_allowed_weights}, "
        f"max_weight_limit {limits.max_weight_limit}"
    )

    for name, miner in miners.items():
        hotkey = miner.hotkey.ss58_address
        landed(
            subtensor.burned_register(miner, netuid),
            f"burned_register {name}",
            lambda hotkey=hotkey: subtensor.is_hotkey_registered(hotkey, netuid=netuid),
        )
    add_stake(subtensor, owner, netuid, 1000, "add_stake for the validator")
    for name, miner in miners.items():
        # Enough alpha for one credit at the published fee, with some to spare.
        # What 100 TAO buys depends on the new subnet's pool, so stake until it is.
        for _ in range(20):
            if stake_on(subtensor, miner, netuid) >= (SUBMISSION_FEE_ALPHA + 5) * 10**9:
                break
            add_stake(subtensor, miner, netuid, 100, f"add_stake {name}")
        else:
            raise SystemExit(f"{name} could not stake {SUBMISSION_FEE_ALPHA} alpha for a credit")
    log("waiting for an epoch so the validator holds a permit")
    for attempt in range(60):
        view = view_of(subtensor, netuid)
        if view.has_validator_permit(owner.hotkey.ss58_address):
            break
        if attempt == 59:
            raise SystemExit("the validator never received a permit")
        wait_blocks(subtensor, 5)
    log(f"neurons: {dict(zip(view.uids, view.hotkeys, strict=True))}")

    # --- 2. local services: a hub and a labeller ---------------------------
    hub = FakeHub(netuid=netuid, key_id="localnet", signing_seed=secrets.token_hex(32))
    hub.register(owner.hotkey.ss58_address, CallerRole.VALIDATOR)
    hub_server = serve(hub, port=args.hub_port)
    labeller = ThreadingHTTPServer(("127.0.0.1", args.labeller_port), Labeller)
    threading.Thread(target=labeller.serve_forever, name="labeller", daemon=True).start()

    shared: dict[str, dict[str, Any]] = {
        "chain": {"network": args.endpoint, "netuid": netuid},
        "hub": {"base_url": f"http://127.0.0.1:{args.hub_port}", "epoch_keys": hub.epoch_keys},
        # The fee is not a setting: one credit is the published 50 alpha.
        "credits": {"treasury_coldkey": treasury},
    }
    configs = {
        name: write_config(
            args.workdir / f"{name}.toml",
            **shared,
            wallet={"name": name},
            miner={"hf_repo": MODELS[name][0], "hf_revision": MODELS[name][1]},
        )
        for name in miners
    }
    configs["owner"] = write_config(
        args.workdir / "owner.toml",
        **shared,
        wallet={"name": "owner"},
        labelling={
            "model": "local-labeller",
            "base_url": f"http://127.0.0.1:{args.labeller_port}/v1",
            "api_key_env": "LOCAL_LABELLER_KEY",
            "temperature": 0.0,
        },
        # One batch per model: on a CPU the policy prefix is the expensive part,
        # and it is computed once per batch.
        evaluation={"device": "cpu", "torch_dtype": "float32", "batch_size": 50},
        # The day holds a few dozen items, so the floor of 100 would rank no
        # model. Everything else in [scoring] stays as shipped.
        scoring={"model_min_scored_items": MIN_SCORED_ITEMS},
        validator={
            "state_directory": str(args.workdir / "state"),
            "submit_weights": True,
            "dry_run": False,
            "submit_results": True,
        },
    )

    # --- 3. miners pay for a credit and commit their models ----------------
    for name, miner in miners.items():
        log(f"{name}: buying one credit")
        cli(home, "credits", "buy", "--config", str(configs[name]), "--yes")
        held = subtensor.get_stake(
            coldkey_ss58=treasury, hotkey_ss58=miner.hotkey.ss58_address, netuid=netuid
        )
        log(f"{name}: the treasury now holds {held} on {name}'s hotkey")
        if held.rao <= 0:
            raise SystemExit("the credit transfer never reached the treasury")
        log(f"{name}: committing {MODELS[name][0]}@{MODELS[name][1][:12]}")
        cli(home, "model", "commit", "--config", str(configs[name]))
        cli(home, "model", "verify", "--config", str(configs[name]))

    # --- 4. the hub freezes yesterday --------------------------------------
    day = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    frozen = subtensor.get_current_block()
    view = view_of(subtensor, netuid)
    groups = {miners[name].hotkey.ss58_address: counts for name, counts in GROUPS.items()}
    epoch_miners = []
    for uid, hotkey in zip(view.uids, view.hotkeys, strict=True):
        verified, members = groups.get(hotkey, (0, 0))
        epoch_miners.append(
            {
                "hotkey": hotkey,
                "uid": uid,
                "coldkey": view.coldkey_for(hotkey),
                "verified_active_members": verified,
                "members": members,
                "eligible": verified >= 10 and members >= 30,
            }
        )
    models = []
    with HuggingFaceClient() as huggingface:
        for name, miner in miners.items():
            repo, revision = MODELS[name]
            metadata = subtensor.get_commitment_metadata(
                netuid=netuid, hotkey_ss58=miner.hotkey.ss58_address
            )
            models.append(
                {
                    "hotkey": miner.hotkey.ss58_address,
                    "repo": repo,
                    "revision": revision,
                    "commit_block": int(metadata["block"]),
                    "fingerprint": weights_fingerprint(huggingface.list_files(repo, revision)),
                    # The treasury holds the fee, as checked above. Deciding
                    # this from the ledger is the real hub's job.
                    "payment": "paid",
                }
            )
    models.sort(key=lambda model: (model["commit_block"], model["hotkey"]))

    truth: dict[str, bool] = {}
    test_lines = []
    for offset, (name, miner) in enumerate(miners.items()):
        for index in range(5):
            for violating, pool in ((True, VIOLATING), (False, COMPLIANT)):
                body = pool[(offset * 5 + index) % len(pool)]
                item_id = f"t_{name[-1]}{index}{'v' if violating else 'c'}"
                truth[item_id] = violating
                test_lines.append(
                    {
                        "id": item_id,
                        "miner_hotkey": miner.hotkey.ss58_address,
                        "author": f"anon-{name[-1]}{index}",
                        "claimed_violating": violating,
                        "body": body,
                        "normalized_sha256": normalized_sha256(body),
                        "created_at": f"{day.isoformat()}T{10 + index:02d}:{offset:02d}:00Z",
                        "verified": True,
                    }
                )
    production_lines = []
    for index, (body, violating) in enumerate(PRODUCTION):
        truth[f"p_{index}"] = violating
        production_lines.append(
            {
                "id": f"p_{index}",
                "body": body,
                "created_at": f"{day.isoformat()}T08:{index:02d}:00Z",
            }
        )
    Labeller.truth = truth

    policy = _policy_text(argparse.Namespace(policy=str(ROOT / "content_policy.md")))
    hub.publish_day(
        day,
        miners=epoch_miners,
        models=models,
        test_lines=test_lines,
        production_lines=production_lines,
        policy_version=policy.version,
        policy_sha256=policy.sha256,
        frozen_block=frozen,
    )
    log(f"epoch {day} frozen at block {frozen}: {len(epoch_miners)} neurons, {len(models)} models")

    # --- 5. the validator scores the day and sets weights ------------------
    started = time.monotonic()
    cli(home, "validator", "run", "--config", str(configs["owner"]), "--date", day.isoformat())
    log(f"validator run took {time.monotonic() - started:.0f}s")

    # --- 6. read back and compare ------------------------------------------
    state = json.loads((args.workdir / "state" / "last-weights.json").read_text())
    record = json.loads(hub.results[(day, owner.hotkey.ss58_address)])
    sent = {hotkey: int(units) for hotkey, units in state["sent_weights"].items()}
    owner_hotkey = subtensor.get_subnet_owner_hotkey(netuid=netuid)

    published = {row["hotkey"]: row["weight"] for row in record["results"] if row["weight"]}
    if record["burned_weight"]:
        published[owner_hotkey] = published.get(owner_hotkey, 0) + record["burned_weight"]
    assert published == sent, f"the record {published} does not reproduce the units sent {sent}"

    by_hand, burned = expected_units(
        record, {model["hotkey"]: model["commit_block"] for model in models}
    )
    if burned:
        by_hand[owner_hotkey] = by_hand.get(owner_hotkey, 0) + burned
    view = view_of(subtensor, netuid)
    by_hand = padded_by_hand(
        by_hand,
        view=view,
        validator=owner.hotkey.ss58_address,
        min_allowed_weights=args.min_allowed_weights,
    )
    assert by_hand == sent, f"by hand {by_hand}, sent {sent}"

    vector = to_u16_chain_vector(resolve_allocations(sent, metagraph=view))
    uids, weights = convert_and_normalize_weights_and_uids(list(vector.uids), list(vector.weights))
    expected_on_chain = dict(zip(uids, weights, strict=True))
    validator_uid = view.uid_for(owner.hotkey.ss58_address)
    on_chain = dict(dict(subtensor.weights(netuid=netuid)).get(validator_uid, []))

    for row in record["results"]:
        if row["dataset_submitted_count"]:
            log(
                f"{row['hotkey'][:10]}: S={row['dataset_submitted_count']} "
                f"V_pos={row['dataset_violating_confirmed']} "
                f"V_neg={row['dataset_compliant_confirmed']} "
                f"accuracy={row['raw_accuracy_bp'] / 100:.2f}% over {row['model_items_scored']} "
                f"P={row['model_parameters']} penalty={row['size_penalty_bp']}bp "
                f"score={row['model_score_bp']}bp rank={row['model_rank']} "
                f"status={row['model_status']} units={row['weight']}"
            )
    log(f"burned: {record['burned_weight']}")
    log(f"units sent:      {sent}")
    log(f"u16 vector sent: {dict(zip(vector.uids, vector.weights, strict=True))}")
    log(f"read from chain: {on_chain}")
    assert on_chain == expected_on_chain, f"chain holds {on_chain}, expected {expected_on_chain}"

    hub_server.shutdown()
    labeller.shutdown()
    log("PASS: chain weights = the vector computed, saved and published = the spec by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
