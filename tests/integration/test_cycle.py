"""A whole validator cycle, over fakes, end to end.

Every other test in this repository checks one module. This one checks that the
modules *compose*: that the DB layer's vocabulary reaches the labeller, that
labels reach the corpus, that the corpus reaches the miners, that the results
reach scoring, and that scoring reaches a weight vector the chain would accept.

That seam is where the expensive mistakes live. A field renamed in one module
and read by another type-checks fine and fails at 04:00 UTC, and no unit test
can see it because each side agrees with itself.

Nothing here reaches the network: the DB layer is the in-memory fake, the chain
is a small object with the five accessors the adapter requires, the labeller is
a scripted transport, and each miner's model is an ``httpx.MockTransport``.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair

from prometheon.cli.cycle import build_results, run_cycle, submit_weights
from prometheon.config import (
    ChainConfig,
    ChainNetwork,
    Config,
    DbLayerConfig,
    EvaluationConfig,
    LabellingConfig,
    ScoringConfig,
    WalletConfig,
)
from prometheon.dbclient.auth import CallerRole
from prometheon.dbclient.client import DbClient
from prometheon.dbclient.fake import FakeDbLayer
from prometheon.dbclient.models import (
    EligibleMiner,
    ModelCommitment,
    ProductionContentItem,
)

# Aliased: pytest tries to collect any module-level name starting with `Test`,
# and warns when it cannot. This is a domain model, not a test class.
from prometheon.dbclient.models import TestContentItem as ContentItem
from prometheon.evaluation.engine import ModerationResult, Verdict
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.registry.validation import (
    UNKNOWN_COMMIT_BLOCK,
    InvalidReason,
    MinerValidation,
    ModelRegistry,
)

pytestmark = pytest.mark.integration

DAY = dt.date(2026, 8, 5)
NETUID = 481
REVISION_A = "a" * 40
REVISION_B = "b" * 40
POLICY = "# Policy\n\nVersion: 2026-08-01\n\nAnswer YES if the item violates.\n"

VALIDATOR = Keypair.create_from_uri("//Validator")
MINER_A = Keypair.create_from_uri("//MinerA")
MINER_B = Keypair.create_from_uri("//MinerB")


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


class FakeChain:
    """The accessors ``prometheon.chain.subtensor`` requires, and no more.

    "And no more" is load-bearing: `assert_sdk_compatible` checks this exact set
    at connect time, so a fake that grew an accessor the runtime does not drive
    would hide the day the SDK stopped offering it.
    """

    def __init__(
        self, hotkeys: list[str], *, owner: str, blocks: dict[str, int] | None = None
    ) -> None:
        self._hotkeys = [owner, *hotkeys]
        self._owner = owner
        #: Block per hotkey, as `get_commitment_metadata` reports it. Settles
        #: duplicate-model claims, so a test can make one miner commit first.
        self._blocks = blocks or {}
        self.submitted: list[dict[str, Any]] = []

    def metagraph(self, netuid: int, lite: bool = True) -> Any:
        return SimpleNamespace(
            # Empty, exactly as an unsynced 10.x metagraph presents it. The
            # transposer must prefer the populated parallel arrays over this.
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

    def get_commitment_metadata(self, netuid: int, hotkey_ss58: str) -> Any:
        # Shaped like `bittensor.core.types.CommitmentOfResponse`, which is what
        # 10.5 returns: `deposit`, `block`, `info`.
        return SimpleNamespace(deposit=0, block=self._blocks.get(hotkey_ss58, 500), info=None)

    def get_commitment(self, netuid: int, uid: int) -> str:
        from prometheon.chain.commitment import ModelCommitment as ChainCommitment
        from prometheon.chain.commitment import encode_commitment

        hotkey = self._hotkeys[uid]
        if hotkey == MINER_A.ss58_address:
            return encode_commitment(ChainCommitment(hf_repo="a/guard", hf_revision=REVISION_A))
        if hotkey == MINER_B.ss58_address:
            return encode_commitment(ChainCommitment(hf_repo="b/guard", hf_revision=REVISION_B))
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
        # `mechid` is named rather than swallowed by **kwargs: the adapter
        # detects mechid support by inspecting this signature, and a fake that
        # hides it behind **kwargs fails the submission gate. That is the gate
        # working, but for the wrong reason.
        self.submitted.append(
            {
                "netuid": netuid,
                "uids": uids,
                "weights": weights,
                "version_key": version_key,
                "mechid": mechid,
            }
        )

        return SimpleNamespace(
            success=True, message="Finalized.", error=None, extrinsic_receipt=None
        )


class StubRegistry(ModelRegistry):
    """Eligibility without Hugging Face.

    Subclassed rather than replaced by a protocol so the cycle keeps calling the
    real type: if `ModelRegistry.validate` changes shape, this fails to
    construct rather than silently diverging.
    """

    def __init__(self, eligible: set[str]) -> None:
        self._eligible = eligible

    def validate(self, entries: Any) -> list[MinerValidation]:
        results = []
        for entry in entries:
            commitment = entry.commitment
            eligible = bool(commitment) and entry.hotkey in self._eligible
            results.append(
                MinerValidation(
                    uid=entry.uid,
                    hotkey=entry.hotkey,
                    valid=eligible,
                    reason=None if eligible else InvalidReason.COMMITMENT_MISSING,
                    detail="",
                    hf_repo=commitment.hf_repo if commitment else "",
                    hf_revision=commitment.hf_revision if commitment else "",
                    commit_block=entry.commit_block,
                    model_type="qwen2" if eligible else "",
                    weight_bytes=8 * 1024**3 if eligible else 0,
                )
            )
        return results


def _labeller(truth: dict[str, bool]) -> OpenAICompatibleClient:
    """A labeller that answers from a truth table, in the requested schema."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        ids = [line.split('"')[3] for line in user.splitlines() if '"id"' in line]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "labels": [
                                        {"id": item_id, "violates": truth.get(item_id, False)}
                                        for item_id in ids
                                    ]
                                }
                            )
                        },
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
    """A model that answers a fixed fraction of each batch correctly.

    Stands in for a loaded checkpoint rather than for an HTTP endpoint: what
    the cycle now depends on is `load`, `moderate`, `unload`, so that is what
    this provides.
    """

    def __init__(self, accuracy: float) -> None:
        self.accuracy = accuracy
        self.loaded = False
        self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.unloaded = True

    def moderate(self, policy: str, items: Any) -> Any:
        verdicts = []
        for index, (item_id, _content) in enumerate(items):
            # Deterministic: the first `accuracy` fraction of each batch is right.
            correct = index < len(items) * self.accuracy
            truthful = item_id.startswith("t")
            verdicts.append(
                Verdict(item_id=item_id, violates=truthful if correct else not truthful)
            )
        return ModerationResult(
            verdicts=tuple(verdicts),
            prompt_tokens=100 * len(verdicts),
            completion_tokens=len(verdicts),
        )


def _install_engines(
    monkeypatch: pytest.MonkeyPatch, accuracy: dict[str, float]
) -> dict[str, _FakeEngine]:
    """Install fake local evaluation in place of downloading and loading.

    Patched at the cycle's own names, so the cycle is exercised exactly as it
    runs: it still resolves a target, asks for a path, builds an engine, and
    hands it the corpus.
    """
    built: dict[str, _FakeEngine] = {}

    def fake_download(target: Any, **_kwargs: Any) -> str:
        return f"/fake/{target.hotkey}"

    def fake_build(model_path: str, **_kwargs: Any) -> _FakeEngine:
        hotkey = model_path.rsplit("/", 1)[-1]
        engine = _FakeEngine(accuracy.get(hotkey, 1.0))
        built[hotkey] = engine
        return engine

    monkeypatch.setattr("prometheon.cli.cycle.download_checkpoint", fake_download)
    monkeypatch.setattr("prometheon.cli.cycle.build_engine", fake_build)
    return built


@pytest.fixture
def config() -> Config:
    return Config(
        chain=ChainConfig(network=ChainNetwork.TEST, netuid=NETUID),
        wallet=WalletConfig(name="validator", hotkey="default"),
        db=DbLayerConfig(base_url="https://db.invalid"),
        labelling=LabellingConfig(api_key_env="PROMETHEON_TEST_KEY", batch_size=25),
        evaluation=EvaluationConfig(batch_size=50),
        # The corpus here is small, so the production floor would exclude
        # everybody. Lowered rather than inflating the fixture, because the
        # composition is what is under test.
        scoring=ScoringConfig(model_min_scored_items=1, miner_burn_share_bp=4000),
    )


def test_dataset_valid_is_only_correctly_labelled_items() -> None:
    """V rewards accurate labelling of both classes and cannot be gamed.

    A claim the labeller confirms is valid; a mislabelled claim (either
    direction) and an unlabellable item are submitted but not valid. This is
    what stops a miner inflating V by claiming everything violates.
    """
    from prometheon.cli.cycle import _dataset_submissions

    hk = MINER_A.ss58_address
    items = [
        ContentItem(id="a", content="aa", author_hotkey=hk, claimed_violating=True, submitted_at=1),
        ContentItem(
            id="b", content="bb", author_hotkey=hk, claimed_violating=False, submitted_at=1
        ),
        ContentItem(id="c", content="cc", author_hotkey=hk, claimed_violating=True, submitted_at=1),
        ContentItem(
            id="d", content="dd", author_hotkey=hk, claimed_violating=False, submitted_at=1
        ),
        ContentItem(id="e", content="ee", author_hotkey=hk, claimed_violating=True, submitted_at=1),
    ]
    snapshot = SimpleNamespace(
        eligible_miners=[EligibleMiner(hotkey=hk, qualified_user_count=1)],
        test_items=items,
    )
    # a: T==T correct, b: F==F correct, c: T!=F wrong, d: F!=T wrong, e: unlabellable.
    labelled = SimpleNamespace(labels={"a": True, "b": False, "c": False, "d": True, "e": None})

    (submission,) = _dataset_submissions(snapshot, labelled)
    assert submission.submitted_count == 5
    assert submission.valid_count == 2


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("PROMETHEON_TEST_KEY", "test-key")

    test_items = [
        ContentItem(
            id=f"t{n:03d}",
            content=f"a threatening message {n}",
            author_hotkey=(MINER_A if n % 2 == 0 else MINER_B).ss58_address,
            claimed_violating=True,
            submitted_at=1,
        )
        for n in range(20)
    ]
    production_items = [
        ProductionContentItem(id=f"p{n:03d}", content=f"an ordinary post {n}", observed_at=1)
        for n in range(30)
    ]
    miners = [
        EligibleMiner(hotkey=MINER_A.ss58_address, qualified_user_count=500),
        EligibleMiner(hotkey=MINER_B.ss58_address, qualified_user_count=700),
    ]
    commitments = [
        ModelCommitment(
            hotkey=MINER_A.ss58_address,
            hf_repo="a/guard",
            revision_sha=REVISION_A,
            committed_at=1,
            block=100,
        ),
        ModelCommitment(
            hotkey=MINER_B.ss58_address,
            hf_repo="b/guard",
            revision_sha=REVISION_B,
            committed_at=1,
            block=200,
        ),
    ]

    layer = FakeDbLayer(netuid=NETUID)
    layer.register(VALIDATOR.ss58_address, CallerRole.VALIDATOR)
    layer.register(MINER_A.ss58_address, CallerRole.MINER)
    layer.register(MINER_B.ss58_address, CallerRole.MINER)
    layer.seed_day(
        DAY,
        test_items=test_items,
        production_items=production_items,
        eligible_miners=miners,
        model_commitments=commitments,
    )

    # Every test item genuinely violates; production items do not.
    truth = {item.id: True for item in test_items}
    truth.update({item.id: False for item in production_items})

    return {
        "layer": layer,
        "truth": truth,
        "chain": FakeChain(
            [MINER_A.ss58_address, MINER_B.ss58_address], owner=VALIDATOR.ss58_address
        ),
        "eligible": {MINER_A.ss58_address, MINER_B.ss58_address},
    }


def _run(
    config: Config,
    world: dict[str, Any],
    *,
    accuracy: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    _install_engines(monkeypatch, accuracy)
    with (
        DbClient(
            base_url="https://db.invalid",
            netuid=NETUID,
            keypair=VALIDATOR,
            transport=world["layer"].transport,
        ) as db,
        _labeller(world["truth"]) as labeller,
    ):
        return run_cycle(
            config=config,
            day=DAY,
            db=db,
            subtensor=world["chain"],
            registry=StubRegistry(world["eligible"]),
            labeller=labeller,
            policy=POLICY,
            policy_version="2026-08-01",
            now=1_700_000_000,
        )


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


def test_a_whole_cycle_produces_a_submittable_weight_vector(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )

    assert len(result.corpus.items) == 50
    assert len(result.evaluations) == 2
    assert result.split.weights, "the cycle produced no weights at all"
    # The burn hotkey is one entry in the vector, so the whole vector — miners
    # and burn together — always sums to the pool.
    assert sum(result.split.weights.values()) == 1_000_000
    # The burn this fixture configures comes off the top before any miner is
    # paid; a full field splits the rest. Stated in the fixture rather than
    # inherited: the shipped default is 0, and this asserts the overlay works.
    assert result.split.burn_units == 400_000
    miner_units = sum(w for hk, w in result.split.weights.items() if hk != result.split.burn_hotkey)
    assert miner_units == 600_000

    receipt = submit_weights(
        result,
        config=config,
        subtensor=world["chain"],
        wallet=type("W", (), {"hotkey": VALIDATOR})(),
        mechid=0,
    )
    assert receipt == "Finalized."

    sent = world["chain"].submitted[0]
    assert sum(sent["weights"]) == 65_535
    assert len(set(sent["uids"])) == len(sent["uids"])


def test_the_cycle_reads_a_real_commit_block_for_every_miner(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect was that nothing populated this on the path that runs.

    ``_duplicate_losers`` settles competing claims on one revision SHA by
    ascending block. Every production construction of ``MinerEntry`` omitted
    ``commit_block``, so all of them carried the same default and the sort
    collapsed to ascending uid — which the registry's own docstring calls the
    strictly worse attack, because a copycat need only hold a lower uid.

    The unit tests passed throughout: they set ``commit_block`` themselves. So
    this asserts it over the *cycle*, which is the only place the omission
    lived, and it asserts a real block rather than merely "not the default".
    """
    world["chain"]._blocks = {
        MINER_A.ss58_address: 4_242,
        MINER_B.ss58_address: 9_001,
    }
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )

    blocks = {item.hotkey: item.commit_block for item in result.validations}
    assert blocks[MINER_A.ss58_address] == 4_242
    assert blocks[MINER_B.ss58_address] == 9_001
    assert UNKNOWN_COMMIT_BLOCK not in blocks.values()


def test_an_unreadable_commit_block_loses_ties_instead_of_winning_them(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The direction of the fallback is the point.

    A block that cannot be read cannot prove priority. Defaulting it to ``0``
    made it the *earliest* possible commitment, so a miner who could make the
    read fail would win every duplicate contest they entered. It sorts last
    instead, and the cycle keeps running.
    """

    def refuse(netuid: int, hotkey_ss58: str) -> Any:
        raise RuntimeError("substrate said no")

    world["chain"].get_commitment_metadata = refuse
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )

    assert all(item.commit_block == UNKNOWN_COMMIT_BLOCK for item in result.validations)
    assert any("duplicate-model tie" in note for note in result.notes)
    # And the cycle still produced a submittable result rather than aborting.
    assert result.split.weights


def test_labelling_batches_are_deterministic_and_interleaved(
    config: Config, world: dict[str, Any]
) -> None:
    """Two properties the labelling path had neither of.

    Evaluation derives its order from the day's content hash so every validator
    batches identically; labelling took whatever order the DB layer returned and
    concatenated test content ahead of production. That made batching
    validator-dependent, and it put each miner's items in one contiguous run —
    so a successful injection inside a test batch landed almost entirely on the
    miner who authored it.
    """
    from prometheon.cli.cycle import _label_items_for

    with DbClient(
        base_url="https://db.invalid",
        netuid=NETUID,
        keypair=VALIDATOR,
        transport=world["layer"].transport,
    ) as db:
        snapshot = db.fetch_day(DAY)

    first = [item.item_id for item in _label_items_for(snapshot)]
    second = [item.item_id for item in _label_items_for(snapshot)]
    assert first == second, "two validators must split the same day into the same calls"

    priors = [item.expected_violating for item in _label_items_for(snapshot)]
    assert set(priors) == {True, False}, "the fixture needs both halves to prove anything"
    # Concatenated order is one run of True followed by one run of False: exactly
    # two transitions' worth of structure. Interleaving destroys that.
    runs = sum(1 for a, b in itertools.pairwise(priors) if a != b)
    assert runs > 1, f"test and production are still contiguous ({runs} transition)"


def test_the_better_model_outranks_the_worse_one(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subnet's core claim, asserted once over the real pipeline."""
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.5},
        monkeypatch=monkeypatch,
    )

    ranked = [score.hotkey for score in result.ranking.scores]
    assert ranked[0] == MINER_A.ss58_address
    assert result.split.weights[MINER_A.ss58_address] > result.split.weights[MINER_B.ss58_address]


def test_a_miner_is_never_scored_on_its_own_test_content(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Self-exclusion, observed through the corpus each miner actually saw.

    Ten of the twenty test items are miner A's, so A must be scored on forty
    items and B on the other forty.
    """
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0},
        monkeypatch=monkeypatch,
    )

    by_hotkey = {item.hotkey: item for item in result.evaluations}
    assert by_hotkey[MINER_A.ss58_address].total == 40
    assert by_hotkey[MINER_B.ss58_address].total == 40

    a_items = {item.item_id for item in result.corpus.for_miner(MINER_A.ss58_address)}
    assert not any(
        item.item_id in a_items
        for item in result.corpus.items
        if item.contributor_hotkey == MINER_A.ss58_address
    )


def test_the_published_record_describes_the_weights_that_were_sent(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record that disagrees with the vector is worse than no record."""
    result = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 0.6},
        monkeypatch=monkeypatch,
    )
    rows = build_results(result)

    assert {row.hotkey for row in rows} == {MINER_A.ss58_address, MINER_B.ss58_address}
    for row in rows:
        assert row.weight == result.split.weights.get(row.hotkey, 0)
        assert row.model_status.value == "evaluated"
        assert row.dataset_submitted_count == 10
        assert row.dataset_valid_count == 10


def test_the_corpus_hash_is_stable_across_two_identical_runs(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two validators with the same data must publish the same corpus hash."""
    first = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0},
        monkeypatch=monkeypatch,
    )
    second = _run(
        config,
        world,
        accuracy={MINER_A.ss58_address: 1.0, MINER_B.ss58_address: 1.0},
        monkeypatch=monkeypatch,
    )
    assert first.corpus_hash == second.corpus_hash
    assert first.split.weights == second.split.weights


def test_a_model_that_cannot_be_run_does_not_stop_the_cycle(
    config: Config, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One miner's unusable checkpoint must cost that miner, and nobody else.

    Every validator hits the same failure at the same time, so an exception
    escaping here would take the whole subnet's submission down over one
    miner's repository.
    """

    class _BrokenEngine(_FakeEngine):
        def load(self) -> None:
            raise RuntimeError("CUDA out of memory")

    def fake_build(model_path: str, **_kwargs: Any) -> _FakeEngine:
        hotkey = model_path.rsplit("/", 1)[-1]
        if hotkey == MINER_B.ss58_address:
            return _BrokenEngine(1.0)
        return _FakeEngine(1.0)

    monkeypatch.setattr(
        "prometheon.cli.cycle.download_checkpoint", lambda target, **_k: f"/fake/{target.hotkey}"
    )
    monkeypatch.setattr("prometheon.cli.cycle.build_engine", fake_build)

    with (
        DbClient(
            base_url="https://db.invalid",
            netuid=NETUID,
            keypair=VALIDATOR,
            transport=world["layer"].transport,
        ) as db,
        _labeller(world["truth"]) as labeller,
    ):
        result = run_cycle(
            config=config,
            day=DAY,
            db=db,
            subtensor=world["chain"],
            registry=StubRegistry(world["eligible"]),
            labeller=labeller,
            policy=POLICY,
            policy_version="2026-08-01",
            now=1_700_000_000,
        )

    by_hotkey = {item.hotkey: item for item in result.evaluations}
    assert by_hotkey[MINER_B.ss58_address].has_token_reading is False
    assert by_hotkey[MINER_A.ss58_address].correct == 40

    # A still earns; B earns nothing from the model half.
    assert result.split.weights[MINER_A.ss58_address] > 0
    rows = {row.hotkey: row for row in build_results(result)}
    assert rows[MINER_B.ss58_address].model_status.value == "unreachable"
