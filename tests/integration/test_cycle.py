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
            return encode_commitment(
                ChainCommitment(hf_repo="a/guard", hf_revision=REVISION_A, chute_id="chute-a")
            )
        if hotkey == MINER_B.ss58_address:
            return encode_commitment(
                ChainCommitment(hf_repo="b/guard", hf_revision=REVISION_B, chute_id="chute-b")
            )
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
    """Eligibility without Hugging Face or Chutes.

    Subclassed rather than replaced by a protocol so the cycle keeps calling the
    real type: if `ModelRegistry.validate` changes shape, this fails to
    construct rather than silently diverging.
    """

    def __init__(self, endpoints: dict[str, str]) -> None:
        self._endpoints = endpoints

    def validate(self, entries: Any) -> list[MinerValidation]:
        results = []
        for entry in entries:
            endpoint = self._endpoints.get(entry.hotkey, "")
            commitment = entry.commitment
            eligible = bool(endpoint and commitment)
            results.append(
                MinerValidation(
                    uid=entry.uid,
                    hotkey=entry.hotkey,
                    valid=eligible,
                    reason=None if eligible else InvalidReason.CHUTE_NOT_RUNNING,
                    detail="",
                    hf_repo=commitment.hf_repo if commitment else "",
                    hf_revision=commitment.hf_revision if commitment else "",
                    chute_id=commitment.chute_id if commitment else "",
                    commit_block=entry.commit_block,
                    chute_slug="prometheon-slug",
                    endpoint_url=endpoint,
                    wrapper_digest="0" * 64,
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


def _models(accuracy: dict[str, float]) -> httpx.Client:
    """Each miner's `/moderate`, answering correctly a fixed fraction of the time."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        body = json.loads(request.content)
        rate = accuracy.get(host, 1.0)
        verdicts = []
        for index, item in enumerate(body["items"]):
            # Deterministic: the first `rate` fraction of each batch is right.
            correct = index < len(body["items"]) * rate
            truthful = item["id"].startswith("t")
            verdicts.append({"id": item["id"], "violates": truthful if correct else not truthful})
        return httpx.Response(
            200,
            json={
                "wrapper_version": "prometheon-moderation/1",
                "policy_version": body["policy_version"],
                "verdicts": verdicts,
                "usage": {"prompt_tokens": 100 * len(verdicts), "completion_tokens": len(verdicts)},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


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
        scoring=ScoringConfig(model_min_scored_items=1),
    )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("PROMETHEON_TEST_KEY", "test-key")

    test_items = [
        ContentItem(
            id=f"t{n:03d}",
            content=f"a threatening message {n}",
            author_hotkey=(MINER_A if n % 2 == 0 else MINER_B).ss58_address,
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
            chute_id="chute-a",
            committed_at=1,
            block=100,
        ),
        ModelCommitment(
            hotkey=MINER_B.ss58_address,
            hf_repo="b/guard",
            revision_sha=REVISION_B,
            chute_id="chute-b",
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
        "endpoints": {
            MINER_A.ss58_address: "https://a.chutes.invalid",
            MINER_B.ss58_address: "https://b.chutes.invalid",
        },
    }


def _run(config: Config, world: dict[str, Any], *, accuracy: dict[str, float]) -> Any:
    with (
        DbClient(
            base_url="https://db.invalid",
            netuid=NETUID,
            keypair=VALIDATOR,
            transport=world["layer"].transport,
        ) as db,
        _labeller(world["truth"]) as labeller,
        _models(accuracy) as evaluation_client,
    ):
        return run_cycle(
            config=config,
            day=DAY,
            db=db,
            subtensor=world["chain"],
            registry=StubRegistry(world["endpoints"]),
            labeller=labeller,
            evaluation_client=evaluation_client,
            policy=POLICY,
            policy_version="2026-08-01",
            chutes_api_key="cpk_test",
            now=1_700_000_000,
        )


# ---------------------------------------------------------------------------
# The cycle
# ---------------------------------------------------------------------------


def test_a_whole_cycle_produces_a_submittable_weight_vector(
    config: Config, world: dict[str, Any]
) -> None:
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 0.6})

    assert len(result.corpus.items) == 50
    assert len(result.evaluations) == 2
    assert result.split.weights, "the cycle produced no weights at all"
    # The burn hotkey is one entry in the vector, so the whole vector — miners
    # and burn together — always sums to the pool.
    assert sum(result.split.weights.values()) == 1_000_000
    # 60% of miner emission burns to the owner before any miner is paid; a full
    # field splits the remaining 40%.
    assert result.split.burn_units == 600_000
    miner_units = sum(
        w for hk, w in result.split.weights.items() if hk != result.split.burn_hotkey
    )
    assert miner_units == 400_000

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
    config: Config, world: dict[str, Any]
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
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 0.6})

    blocks = {item.hotkey: item.commit_block for item in result.validations}
    assert blocks[MINER_A.ss58_address] == 4_242
    assert blocks[MINER_B.ss58_address] == 9_001
    assert UNKNOWN_COMMIT_BLOCK not in blocks.values()


def test_an_unreadable_commit_block_loses_ties_instead_of_winning_them(
    config: Config, world: dict[str, Any]
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
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 0.6})

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


def test_the_better_model_outranks_the_worse_one(config: Config, world: dict[str, Any]) -> None:
    """The subnet's core claim, asserted once over the real pipeline."""
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 0.5})

    ranked = [score.hotkey for score in result.ranking.scores]
    assert ranked[0] == MINER_A.ss58_address
    assert result.split.weights[MINER_A.ss58_address] > result.split.weights[MINER_B.ss58_address]


def test_a_miner_is_never_scored_on_its_own_test_content(
    config: Config, world: dict[str, Any]
) -> None:
    """Self-exclusion, observed through the corpus each miner actually saw.

    Ten of the twenty test items are miner A's, so A must be scored on forty
    items and B on the other forty.
    """
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 1.0})

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
    config: Config, world: dict[str, Any]
) -> None:
    """A record that disagrees with the vector is worse than no record."""
    result = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 0.6})
    rows = build_results(result)

    assert {row.hotkey for row in rows} == {MINER_A.ss58_address, MINER_B.ss58_address}
    for row in rows:
        assert row.weight == result.split.weights.get(row.hotkey, 0)
        assert row.model_status.value == "evaluated"
        assert row.dataset_submitted_count == 10
        assert row.dataset_valid_count == 10


def test_the_corpus_hash_is_stable_across_two_identical_runs(
    config: Config, world: dict[str, Any]
) -> None:
    """Two validators with the same data must publish the same corpus hash."""
    first = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 1.0})
    second = _run(config, world, accuracy={"a.chutes.invalid": 1.0, "b.chutes.invalid": 1.0})
    assert first.corpus_hash == second.corpus_hash
    assert first.split.weights == second.split.weights


def test_an_unreachable_model_does_not_stop_the_cycle(
    config: Config, world: dict[str, Any]
) -> None:
    """One miner's dead deployment must cost that miner, and nobody else."""

    def half_dead(request: httpx.Request) -> httpx.Response:
        if request.url.host == "b.chutes.invalid":
            raise httpx.ConnectError("connection refused")
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "wrapper_version": "prometheon-moderation/1",
                "policy_version": body["policy_version"],
                "verdicts": [
                    {"id": item["id"], "violates": item["id"].startswith("t")}
                    for item in body["items"]
                ],
                "usage": {
                    "prompt_tokens": 100 * len(body["items"]),
                    "completion_tokens": len(body["items"]),
                },
            },
        )

    with (
        DbClient(
            base_url="https://db.invalid",
            netuid=NETUID,
            keypair=VALIDATOR,
            transport=world["layer"].transport,
        ) as db,
        _labeller(world["truth"]) as labeller,
        httpx.Client(transport=httpx.MockTransport(half_dead)) as evaluation_client,
    ):
        result = run_cycle(
            config=config,
            day=DAY,
            db=db,
            subtensor=world["chain"],
            registry=StubRegistry(world["endpoints"]),
            labeller=labeller,
            evaluation_client=evaluation_client,
            policy=POLICY,
            policy_version="2026-08-01",
            chutes_api_key="cpk_test",
            now=1_700_000_000,
        )

    by_hotkey = {item.hotkey: item for item in result.evaluations}
    assert by_hotkey[MINER_B.ss58_address].has_token_reading is False
    assert by_hotkey[MINER_A.ss58_address].correct == 40

    # A still earns; B earns nothing from the model half.
    assert result.split.weights[MINER_A.ss58_address] > 0
    rows = {row.hotkey: row for row in build_results(result)}
    assert rows[MINER_B.ss58_address].model_status.value == "unreachable"
