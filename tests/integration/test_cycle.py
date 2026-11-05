"""A whole validator cycle, over fakes, end to end.

Every other test in this repository checks one module. This one checks that the
modules *compose*: that the hub's signed epoch reaches eligibility, that feed
lines reach near-duplicate removal and the labeller, that labels reach the
corpus, that the corpus reaches the miners, that the results reach scoring, and
that scoring reaches a weight vector the chain would accept.

Nothing here reaches the network: the hub is the in-memory fake, signing real
epochs; the chain is a small object with the accessors the adapter requires;
the labeller is a scripted transport; and each miner's model is a test double
that answers a fixed fraction of each batch correctly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair

from prometheon.cli.cycle import Policy, build_results, run_cycle, submit_weights
from prometheon.config import (
    ChainConfig,
    ChainNetwork,
    Config,
    EvaluationConfig,
    HubConfig,
    LabellingConfig,
    ScoringConfig,
    WalletConfig,
)
from prometheon.errors import ConfigError, ValidatorFaultError
from prometheon.evaluation.engine import ModerationResult, Verdict
from prometheon.hub.auth import CallerRole
from prometheon.hub.client import HubClient
from prometheon.hub.fake import FakeHub
from prometheon.hub.models import ModelStatus
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.registry.validation import InvalidReason, MinerValidation, ModelRegistry
from prometheon.scoring.dedupe import normalized_sha256

pytestmark = pytest.mark.integration

DAY = dt.date(2026, 11, 5)
NOW = int(dt.datetime(2026, 11, 6, 4, 0, tzinfo=dt.timezone.utc).timestamp())
NETUID = 481
POLICY_TEXT = "# Policy\n\nVersion: 2026-08-01\n\nAnswer YES if the item violates.\n"
POLICY = Policy(
    text=POLICY_TEXT,
    version="2026-08-01",
    sha256=hashlib.sha256(POLICY_TEXT.encode()).hexdigest(),
)

VALIDATOR = Keypair.create_from_uri("//Validator")
MINER_A = Keypair.create_from_uri("//MinerA")
MINER_B = Keypair.create_from_uri("//MinerB")
MINER_C = Keypair.create_from_uri("//MinerC")
COLD_A = Keypair.create_from_uri("//ColdA")
COLD_B = Keypair.create_from_uri("//ColdB")
COLD_C = Keypair.create_from_uri("//ColdC")

_WORDS = (
    "river amber quartz meadow lantern orbit velvet cobalt harbor timber falcon "
    "pepper glacier saddle violet ember canyon marble thistle walnut copper "
    "drizzle beacon juniper pebble saffron tundra willow zephyr cinder hollow"
).split()


def _text(seed: int, *, violating: bool) -> str:
    """A post no other fixture post is a near-duplicate of.

    Words are drawn by hash rather than by stride: a stride through a short
    word list repeats after a few steps, and those posts really are
    near-duplicates of each other.
    """
    words = " ".join(
        _WORDS[hashlib.sha256(f"{seed}:{step}".encode()).digest()[0] % len(_WORDS)]
        for step in range(14)
    )
    return f"{'THREAT' if violating else 'KIND'} {seed}: {words}"


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


class FakeChain:
    """The accessors ``prometheon.chain.subtensor`` requires, and no more."""

    def __init__(self, hotkeys: list[str], *, owner: str) -> None:
        self._hotkeys = [owner, *hotkeys]
        self._owner = owner
        self.submitted: list[dict[str, Any]] = []

    def metagraph(self, netuid: int, lite: bool = True) -> Any:
        return SimpleNamespace(
            neurons=[],
            hotkeys=list(self._hotkeys),
            uids=list(range(len(self._hotkeys))),
            validator_permit=[True] + [False] * (len(self._hotkeys) - 1),
            stake=[1.0] * len(self._hotkeys),
            block=1_000,
        )

    def get_subnet_hyperparameters(self, netuid: int) -> Any:
        return SimpleNamespace(weights_version=0, commit_reveal_weights_enabled=False)

    def get_subnet_owner_hotkey(self, netuid: int) -> str:
        return self._owner

    def get_commitment(self, netuid: int, uid: int) -> str:
        # Validators no longer read commitments from chain; the epoch freezes
        # them. Still required, because `model commit` and `model verify` do.
        return ""

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
        self.submitted.append({"netuid": netuid, "uids": uids, "weights": weights})
        return SimpleNamespace(
            success=True, message="Finalized.", error=None, extrinsic_receipt=None
        )


class StubRegistry(ModelRegistry):
    """Eligibility without Hugging Face: every entry it is handed is valid."""

    def __init__(
        self,
        invalid: set[str] | None = None,
        reason: InvalidReason = InvalidReason.MANIFEST_VIOLATION,
    ) -> None:
        self._invalid = invalid or set()
        self._reason = reason

    def validate(self, entries: Any) -> list[MinerValidation]:
        self.seen = [entry.hotkey for entry in entries]
        return [
            MinerValidation(
                uid=entry.uid,
                hotkey=entry.hotkey,
                valid=entry.hotkey not in self._invalid,
                reason=self._reason if entry.hotkey in self._invalid else None,
                detail="",
                hf_repo=entry.commitment.hf_repo,
                hf_revision=entry.commitment.hf_revision,
                commit_block=entry.commit_block,
                model_type="qwen2",
                weight_bytes=8 * 1024**3,
            )
            for entry in entries
        ]


def _labeller(truth: dict[str, bool]) -> OpenAICompatibleClient:
    """A labeller that answers from a truth table, in the requested schema."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        ids = [line.split('"')[3] for line in user.splitlines() if '"id"' in line]
        labels = [{"id": item_id, "violates": truth.get(item_id, False)} for item_id in ids]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps({"labels": labels})},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    return OpenAICompatibleClient(
        LabellingConfig(api_key_env="PROMETHEON_TEST_KEY"),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )


class _FakeEngine:
    """A model that answers a fixed fraction of each batch correctly."""

    def __init__(self, accuracy: float, *, broken: bool = False) -> None:
        self.accuracy = accuracy
        self.broken = broken
        self.seen: list[str] = []

    def load(self) -> None:
        if self.broken:
            raise RuntimeError("the checkpoint is corrupt")

    def unload(self) -> None:
        return None

    def moderate(self, policy: str, items: Any) -> Any:
        verdicts = []
        for index, (item_id, content) in enumerate(items):
            self.seen.append(item_id)
            correct = index < len(items) * self.accuracy
            truthful = content.startswith("THREAT")
            verdicts.append(
                Verdict(item_id=item_id, violates=truthful if correct else not truthful)
            )
        return ModerationResult(
            verdicts=tuple(verdicts),
            prompt_tokens=100 * len(verdicts),
            completion_tokens=len(verdicts),
        )


def _install_engines(
    monkeypatch: pytest.MonkeyPatch,
    accuracy: dict[str, float],
    broken: frozenset[str] = frozenset(),
) -> dict[str, _FakeEngine]:
    built: dict[str, _FakeEngine] = {}

    def fake_download(target: Any, **_kwargs: Any) -> str:
        return f"/fake/{target.hotkey}"

    def fake_build(model_path: str, **_kwargs: Any) -> _FakeEngine:
        hotkey = model_path.rsplit("/", 1)[-1]
        built[hotkey] = _FakeEngine(accuracy.get(hotkey, 1.0), broken=hotkey in broken)
        return built[hotkey]

    monkeypatch.setattr("prometheon.cli.cycle.download_checkpoint", fake_download)
    monkeypatch.setattr("prometheon.cli.cycle.build_engine", fake_build)
    return built


def _miner(key: Keypair, uid: int, cold: Keypair, members: int) -> dict[str, Any]:
    return {
        "hotkey": key.ss58_address,
        "uid": uid,
        "coldkey": cold.ss58_address,
        "verified_active_members": members,
        "eligible": members >= 10,
    }


def _model(key: Keypair, repo: str, block: int, payment: str = "paid") -> dict[str, Any]:
    return {
        "hotkey": key.ss58_address,
        "repo": repo,
        "revision": hashlib.sha1(repo.encode()).hexdigest(),  # noqa: S324 - a fixture id
        "commit_block": block,
        "fingerprint": hashlib.sha256(repo.encode()).hexdigest(),
        "payment": payment,
    }


def _test_lines(key: Keypair, prefix: str, count: int, start_seed: int) -> list[dict[str, Any]]:
    """Half claimed violating and half claimed compliant, every claim true."""
    lines = []
    for index in range(count):
        violating = index % 2 == 0
        body = _text(start_seed + index, violating=violating)
        lines.append(
            {
                "id": f"{prefix}{index:02d}",
                "miner_hotkey": key.ss58_address,
                "author": f"anon-{prefix}{index}",
                "claimed_violating": violating,
                "body": body,
                "normalized_sha256": normalized_sha256(body),
                "created_at": f"2026-11-05T{10 + index // 6:02d}:{(index * 7) % 60:02d}:00Z",
                "verified": True,
            }
        )
    return lines


def _production_lines(count: int) -> list[dict[str, Any]]:
    return [
        {
            "id": f"p{index:02d}",
            "body": _text(500 + index, violating=False),
            "created_at": "2026-11-05T08:00:00Z",
        }
        for index in range(count)
    ]


@pytest.fixture
def config() -> Config:
    return Config(
        chain=ChainConfig(network=ChainNetwork.TEST, netuid=NETUID),
        wallet=WalletConfig(name="validator", hotkey="default"),
        hub=HubConfig(base_url="https://hub.invalid"),
        labelling=LabellingConfig(api_key_env="PROMETHEON_TEST_KEY", batch_size=25),
        evaluation=EvaluationConfig(batch_size=50),
        # The corpus here is small, so the default measurement floor of 100
        # would rank nobody. Lowered rather than inflating the fixture, because
        # the composition is what is under test.
        scoring=ScoringConfig(model_min_scored_items=1),
    )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Two eligible miners with models, one ineligible one, and production content."""
    monkeypatch.setenv("PROMETHEON_TEST_KEY", "test-key")
    hub = FakeHub(netuid=NETUID, now=lambda: NOW)
    hub.register(VALIDATOR.ss58_address, CallerRole.VALIDATOR)
    tests = [
        *_test_lines(MINER_A, "ta", 10, 0),
        *_test_lines(MINER_B, "tb", 10, 100),
        *_test_lines(MINER_C, "tc", 4, 200),
    ]
    productions = _production_lines(30)
    hub.publish_day(
        DAY,
        miners=[
            _miner(VALIDATOR, 0, VALIDATOR, 0),
            _miner(MINER_A, 1, COLD_A, 14),
            _miner(MINER_B, 2, COLD_B, 14),
            # Below the floor: its content is still verified human content and
            # enters the corpus, but it earns from neither pool.
            _miner(MINER_C, 3, COLD_C, 3),
        ],
        models=[
            _model(MINER_A, "a/guard", 100),
            _model(MINER_B, "b/guard", 200),
            _model(MINER_C, "c/guard", 300),
        ],
        test_lines=tests,
        production_lines=productions,
        policy_version=POLICY.version,
        policy_sha256=POLICY.sha256,
    )
    truth = {line["id"]: line["claimed_violating"] for line in tests}
    truth.update({line["id"]: False for line in productions})
    return {
        "hub": hub,
        "truth": truth,
        "tests": tests,
        "productions": productions,
        "chain": FakeChain(
            [MINER_A.ss58_address, MINER_B.ss58_address, MINER_C.ss58_address],
            owner=VALIDATOR.ss58_address,
        ),
    }


def _run(
    config: Config,
    world: dict[str, Any],
    *,
    accuracy: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
    registry: ModelRegistry | None = None,
    policy: Policy = POLICY,
    broken: frozenset[str] = frozenset(),
    check_runtime: Any = None,
) -> Any:
    _install_engines(monkeypatch, accuracy, broken)
    with (
        HubClient(
            base_url="https://hub.invalid",
            netuid=NETUID,
            epoch_keys=world["hub"].epoch_keys,
            keypair=VALIDATOR,
            transport=world["hub"].transport,
            now=lambda: NOW,
        ) as hub,
        _labeller(world["truth"]) as labeller,
    ):
        return run_cycle(
            config=config,
            day=DAY,
            hub=hub,
            subtensor=world["chain"],
            registry=registry or StubRegistry(),
            labeller=labeller,
            policy=policy,
            now=NOW,
            # The real check builds a model on the configured device, which
            # needs the evaluation extra CI does not install. The tests that
            # are about it pass their own.
            check_runtime=check_runtime or (lambda: None),
        )


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


def test_a_whole_cycle_produces_the_exact_v3_vector(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )

    assert len(result.corpus.items) == 24 + 30
    # A and B each score 2 * min(5, 5) * 10/10 = 10 and split 70%. C is below
    # the floor. A is the better model and takes 25%, B 4%, and the empty third
    # rank's 1% burns to the owner.
    assert dict(result.split.weights) == {
        MINER_A.ss58_address: 350_000 + 250_000,
        MINER_B.ss58_address: 350_000 + 40_000,
        VALIDATOR.ss58_address: 10_000,
    }

    sent = submit_weights(
        result,
        config=config,
        subtensor=world["chain"],
        wallet=SimpleNamespace(hotkey=VALIDATOR),
        mechid=0,
    )
    assert sent.receipt == "Finalized."
    assert sent.gone == ()
    assert dict(sent.units) == dict(result.split.weights)
    on_chain = world["chain"].submitted[0]
    assert sum(on_chain["weights"]) == 65_535
    assert on_chain["uids"] == [0, 1, 2]


def test_only_paid_models_of_eligible_miners_reach_the_registry(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubRegistry()
    result = _run(
        config,
        world,
        accuracy={},
        monkeypatch=monkeypatch,
        registry=registry,
    )
    assert registry.seen == [MINER_A.ss58_address, MINER_B.ss58_address]
    assert result.model_statuses == {MINER_C.ss58_address: ModelStatus.INELIGIBLE}


def test_unpaid_and_duplicate_models_are_listed_but_not_evaluated(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub: FakeHub = world["hub"]
    hub.publish_day(
        DAY,
        miners=[
            _miner(VALIDATOR, 0, VALIDATOR, 0),
            _miner(MINER_A, 1, COLD_A, 14),
            _miner(MINER_B, 2, COLD_B, 14),
        ],
        models=[
            _model(MINER_A, "a/guard", 100, "unpaid"),
            _model(MINER_B, "b/guard", 200, "duplicate"),
        ],
        test_lines=world["tests"][:20],
        production_lines=world["productions"],
        policy_version=POLICY.version,
        policy_sha256=POLICY.sha256,
    )
    result = _run(config, world, accuracy={}, monkeypatch=monkeypatch)

    assert not result.evaluations
    rows = {row.hotkey: row for row in build_results(result)}
    assert rows[MINER_A.ss58_address].model_status is ModelStatus.UNPAID
    assert rows[MINER_B.ss58_address].model_status is ModelStatus.DUPLICATE
    # The whole 30% model pool burns; data still pays.
    assert result.split.burn_units == 300_000


def test_a_miner_is_never_scored_on_content_from_its_own_coldkey(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-exclusion by coldkey, not hotkey (B9).

    MINER_C's hotkey is moved under COLD_A: its four items must leave A's view
    too, although they were written for a different hotkey.
    """
    hub: FakeHub = world["hub"]
    hub.publish_day(
        DAY,
        miners=[
            _miner(VALIDATOR, 0, VALIDATOR, 0),
            _miner(MINER_A, 1, COLD_A, 14),
            _miner(MINER_B, 2, COLD_B, 14),
            _miner(MINER_C, 3, COLD_A, 3),
        ],
        models=[_model(MINER_A, "a/guard", 100), _model(MINER_B, "b/guard", 200)],
        test_lines=world["tests"],
        production_lines=world["productions"],
        policy_version=POLICY.version,
        policy_sha256=POLICY.sha256,
    )
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0},
        monkeypatch=monkeypatch,
    )

    by_hotkey = {item.hotkey: item for item in result.evaluations}
    assert by_hotkey[MINER_A.ss58_address].total == 54 - 10 - 4
    assert by_hotkey[MINER_B.ss58_address].total == 54 - 10
    a_view = {
        item.item_id for item in result.corpus.for_miner(MINER_A.ss58_address, COLD_A.ss58_address)
    }
    assert not any(item_id.startswith(("ta", "tc")) for item_id in a_view)


def test_near_duplicates_are_removed_before_labelling_and_cost_the_copier(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """B posts a copy of A's earlier item, reworded only in case and punctuation."""
    hub: FakeHub = world["hub"]
    original = world["tests"][0]
    copy = {
        **world["tests"][10],
        "id": "tb-copy",
        "body": original["body"].upper() + "!!!",
        "created_at": "2026-11-05T23:00:00Z",
        "claimed_violating": original["claimed_violating"],
    }
    copy["normalized_sha256"] = normalized_sha256(copy["body"])
    hub.publish_day(
        DAY,
        miners=[
            _miner(VALIDATOR, 0, VALIDATOR, 0),
            _miner(MINER_A, 1, COLD_A, 14),
            _miner(MINER_B, 2, COLD_B, 14),
            _miner(MINER_C, 3, COLD_C, 3),
        ],
        models=[_model(MINER_A, "a/guard", 100), _model(MINER_B, "b/guard", 200)],
        test_lines=[*world["tests"], copy],
        production_lines=world["productions"],
        policy_version=POLICY.version,
        policy_sha256=POLICY.sha256,
    )
    world["truth"]["tb-copy"] = copy["claimed_violating"]
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0},
        monkeypatch=monkeypatch,
    )

    assert dict(result.duplicates) == {"tb-copy": original["id"]}
    assert "tb-copy" not in result.labelled.labels
    scores = {score.hotkey: score for score in result.dataset_scores}
    assert scores[MINER_B.ss58_address].submitted_count == 10


def test_labelling_batches_are_deterministic_and_interleaved(
    config: Config, world: dict[str, Any]
) -> None:
    from prometheon.cli.cycle import _label_items_for

    with HubClient(
        base_url="https://hub.invalid",
        netuid=NETUID,
        epoch_keys=world["hub"].epoch_keys,
        keypair=VALIDATOR,
        transport=world["hub"].transport,
        now=lambda: NOW,
    ) as hub:
        data = hub.fetch_day(DAY, scoring_version="prometheon-scoring/3.0")

    kept = frozenset(line.item_id for line in (*data.feed.test, *data.feed.production))
    coldkeys = {miner.hotkey: miner.coldkey for miner in data.epoch.miners}
    first = [item.item_id for item in _label_items_for(data, kept, coldkeys)]
    second = [item.item_id for item in _label_items_for(data, kept, coldkeys)]
    assert first == second

    items = _label_items_for(data, kept, coldkeys)
    sources = [item.author is None for item in items]
    runs = sum(1 for a, b in itertools.pairwise(sources) if a != b)
    assert runs > 1, "test and production are still contiguous"
    # The tripwire's author is the coldkey, so two hotkeys of one party are one source.
    assert {item.author for item in items if item.author} == {
        COLD_A.ss58_address,
        COLD_B.ss58_address,
        COLD_C.ss58_address,
    }


def test_the_published_record_reproduces_the_vector(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows carry earnings, the record carries burn, and together they are the vector."""
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )
    rows = build_results(result)

    assert {row.hotkey for row in rows} == {
        MINER_A.ss58_address,
        MINER_B.ss58_address,
        MINER_C.ss58_address,
    }
    rebuilt = {row.hotkey: row.weight for row in rows if row.weight}
    rebuilt[result.burn_hotkey] = rebuilt.get(result.burn_hotkey, 0) + result.split.burn_units
    assert rebuilt == dict(result.split.weights)

    by_hotkey = {row.hotkey: row for row in rows}
    assert by_hotkey[MINER_A.ss58_address].dataset_violating_confirmed == 5
    assert by_hotkey[MINER_A.ss58_address].dataset_compliant_confirmed == 5
    assert by_hotkey[MINER_C.ss58_address].model_status is ModelStatus.INELIGIBLE
    assert by_hotkey[MINER_C.ss58_address].weight == 0


def test_a_policy_the_epoch_did_not_freeze_stops_the_cycle(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    other = Policy(text="different", version="2026-08-01", sha256="0" * 64)
    with pytest.raises(ConfigError, match="froze policy"):
        _run(config, world, accuracy={}, monkeypatch=monkeypatch, policy=other)


def test_rejected_feed_lines_are_reported_and_cost_nobody_else(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    hub: FakeHub = world["hub"]
    unverified = {**world["tests"][1], "id": "ta-unverified", "verified": False}
    hub.publish_day(
        DAY,
        miners=[
            _miner(VALIDATOR, 0, VALIDATOR, 0),
            _miner(MINER_A, 1, COLD_A, 14),
            _miner(MINER_B, 2, COLD_B, 14),
            _miner(MINER_C, 3, COLD_C, 3),
        ],
        models=[_model(MINER_A, "a/guard", 100), _model(MINER_B, "b/guard", 200)],
        test_lines=[*world["tests"], unverified],
        production_lines=world["productions"],
        policy_version=POLICY.version,
        policy_sha256=POLICY.sha256,
    )
    result = _run(config, world, accuracy={}, monkeypatch=monkeypatch)

    assert any("1 feed lines rejected" in note for note in result.notes)
    assert "ta-unverified" not in result.labelled.labels
    scores = {score.hotkey: score for score in result.dataset_scores}
    assert scores[MINER_A.ss58_address].submitted_count == 10


def test_two_identical_runs_agree_exactly(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    accuracy = {MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0}
    first = _run(config, world, accuracy=accuracy, monkeypatch=monkeypatch)
    second = _run(config, world, accuracy=accuracy, monkeypatch=monkeypatch)
    assert first.corpus_hash == second.corpus_hash
    assert first.epoch_content_hash == second.epoch_content_hash
    assert dict(first.split.weights) == dict(second.split.weights)


def test_a_model_that_cannot_be_run_costs_only_its_miner(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One miner's unusable checkpoint must cost that miner, and nobody else."""
    result = _run(
        config,
        world,
        accuracy={},
        monkeypatch=monkeypatch,
        broken=frozenset({MINER_B.ss58_address}),
    )

    by_hotkey = {item.hotkey: item for item in result.evaluations}
    assert by_hotkey[MINER_B.ss58_address].has_token_reading is False
    assert by_hotkey[MINER_A.ss58_address].correct == by_hotkey[MINER_A.ss58_address].total
    rows = {row.hotkey: row for row in build_results(result)}
    assert rows[MINER_B.ss58_address].model_status is ModelStatus.UNREACHABLE
    assert dict(result.split.model_units) == {MINER_A.ss58_address: 250_000}


def test_hugging_face_not_answering_stops_the_cycle_before_labelling(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """B21: an outage on this validator must not score a miner out of the day."""
    registry = StubRegistry({MINER_B.ss58_address}, reason=InvalidReason.UPSTREAM_UNAVAILABLE)
    labelled: list[str] = []
    monkeypatch.setattr(
        "prometheon.cli.cycle.label_items",
        lambda *args, **kwargs: labelled.append("called"),
    )
    with pytest.raises(ValidatorFaultError, match="could not be reached"):
        _run(config, world, accuracy={}, monkeypatch=monkeypatch, registry=registry)
    assert not labelled


def test_a_repository_the_miner_removed_costs_only_that_miner(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = StubRegistry({MINER_B.ss58_address}, reason=InvalidReason.REPO_UNAVAILABLE)
    result = _run(config, world, accuracy={}, monkeypatch=monkeypatch, registry=registry)
    rows = {row.hotkey: row for row in build_results(result)}
    assert rows[MINER_B.ss58_address].model_status is ModelStatus.INVALID
    assert dict(result.split.model_units) == {MINER_A.ss58_address: 250_000}


class TestValidatorFaultsStopTheCycle:
    """B10: a fault on this validator must not score the field zero and submit."""

    def test_a_validator_that_cannot_run_a_model_stops_before_labelling(
        self, config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        labelled: list[str] = []
        monkeypatch.setattr(
            "prometheon.cli.cycle.label_items", lambda *a, **k: labelled.append("called")
        )

        def broken_runtime() -> None:
            raise ValidatorFaultError("the evaluation runtime is not installed")

        with pytest.raises(ValidatorFaultError, match="not installed"):
            _run(config, world, accuracy={}, monkeypatch=monkeypatch, check_runtime=broken_runtime)
        assert not labelled

    def test_a_runtime_that_dies_mid_cycle_stops_the_cycle(
        self, config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The canary runs again after a model fails every batch."""
        checks: list[str] = []

        def runtime() -> None:
            checks.append("check")
            if len(checks) > 1:
                raise ValidatorFaultError("CUDA error: device lost")

        with pytest.raises(ValidatorFaultError, match="device lost"):
            _run(
                config,
                world,
                accuracy={},
                monkeypatch=monkeypatch,
                broken=frozenset({MINER_A.ss58_address}),
                check_runtime=runtime,
            )
        assert len(checks) == 2

    def test_a_broken_model_on_a_healthy_validator_is_scored_not_stopped(
        self, config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checks: list[str] = []
        result = _run(
            config,
            world,
            accuracy={},
            monkeypatch=monkeypatch,
            broken=frozenset({MINER_B.ss58_address}),
            check_runtime=lambda: checks.append("check"),
        )
        assert len(checks) == 2  # before labelling, and after B failed every batch
        assert any(
            MINER_B.ss58_address in note and "every batch failed" in note for note in result.notes
        )
