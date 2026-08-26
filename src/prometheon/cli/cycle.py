"""One validator cycle, end to end.

Separated from argument parsing so it can be driven by a test with fakes rather
than only by a terminal. Every external system arrives as a parameter; nothing
in here opens a connection or reads the environment.

The order of the stages is a contract:

1. **Chain gates first.** Commit-reveal, weights version, and mechid support are
   checked *before* a single API call is paid for. A cycle that cannot submit
   its result must not spend fourteen hours and an OpenAI bill discovering that.
2. **Snapshot, then chain commitments.** The DB layer says who is eligible; the
   *chain* says what they committed. Eligibility to earn and the model being
   scored come from different authorities, so a DB layer that invents a
   commitment cannot get a model scored: the commitment is read from chain and
   verified against Hugging Face independently.
3. **Label, then evaluate.** Ground truth is fixed before any model sees the
   corpus, so no model can influence what it is measured against.
4. **Score, then submit.** Weights and the published record are computed from
   the same in-memory result, so the record always describes the weights that
   were actually sent.

The cycle is a pure function of its inputs given the same external answers, and
:func:`run_cycle` returns everything it computed so a caller can print it, sign
it, or diff two validators' runs against each other.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from prometheon.chain import subtensor as chain

# Two different types are called `ModelCommitment`: the chain codec's (with
# `hf_revision`) and the DB layer's mirrored record (with `revision_sha`). This
# module is the one place both are in scope, so both are aliased. An unaliased
# import here would make a field-name mistake look like a typo rather than a
# type confusion.
from prometheon.chain.commitment import ModelCommitment as ChainCommitment
from prometheon.chain.commitment import read_commitment, read_commitment_block
from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import (
    assert_submission_allowed,
    resolve_allocations,
    to_u16_chain_vector,
)
from prometheon.config import Config
from prometheon.dbclient.client import DaySnapshot, DbClient
from prometheon.dbclient.models import (
    MinerResult,
    ModelStatus,
    corpus_content_hash,
)
from prometheon.dbclient.models import ModelCommitment as FrozenCommitment
from prometheon.errors import CommitmentError, EvaluationError, SubtensorError
from prometheon.evaluation.corpus import (
    EvaluationCorpus,
    ItemSource,
    LabelledItem,
    build_corpus,
    corpus_order_key,
)
from prometheon.evaluation.result import MinerEvaluation
from prometheon.evaluation.runner import (
    MinerTarget,
    build_engine,
    download_checkpoint,
    evaluate_miner,
)
from prometheon.labelling.batch import LabelledCorpus, label_items
from prometheon.labelling.client import ChatCompletionClient
from prometheon.labelling.prompt import LabelItem
from prometheon.registry.validation import (
    UNKNOWN_COMMIT_BLOCK,
    MinerEntry,
    MinerValidation,
    ModelRegistry,
)
from prometheon.scoring.combine import WeightSplit, combine_weights
from prometheon.scoring.dataset import DatasetSubmission, score_dataset
from prometheon.scoring.model import ModelEvaluation, ModelRanking, score_models

#: Bumped when a change here would make two validators compute different
#: weights from identical inputs. Published in every evaluation record so a
#: divergence can be attributed to a version skew rather than to a disagreement.
SCORING_VERSION = "prometheon-scoring/2.1"

_BASIS_POINTS = 10_000


@dataclass(frozen=True)
class CycleResult:
    """Everything one cycle computed. Nothing here has been sent anywhere."""

    day: dt.date
    snapshot: DaySnapshot
    labelled: LabelledCorpus
    corpus: EvaluationCorpus
    validations: tuple[MinerValidation, ...]
    evaluations: tuple[MinerEvaluation, ...]
    ranking: ModelRanking
    split: WeightSplit
    burn_hotkey: str
    metagraph: MetagraphView
    started_at: int
    completed_at: int = 0
    submitted_receipt: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def corpus_hash(self) -> str:
        return corpus_content_hash(
            date=self.day,
            labelled=[(item.item_id, item.violating) for item in self.corpus.items],
        )


def run_cycle(
    *,
    config: Config,
    day: dt.date,
    db: DbClient,
    subtensor: Any,
    registry: ModelRegistry,
    labeller: ChatCompletionClient,
    policy: str,
    policy_version: str,
    now: int,
    progress: Any = None,
) -> CycleResult:
    """Run one day's cycle and return what it computed, without submitting."""
    say = progress if progress is not None else (lambda _message: None)

    # --- 1. can this cycle submit at all? --------------------------------
    say("checking chain submission gates")
    capabilities = chain.detect_capabilities(subtensor)
    hyperparameters = chain.read_hyperparameters(subtensor, netuid=config.chain.netuid)
    assert_submission_allowed(
        hyperparameters=hyperparameters,
        capabilities=capabilities,
        configured_version_key=config.chain.version_key,
        fail_on_weights_version_mismatch=config.chain.fail_on_weights_version_mismatch,
        allow_missing_mechid=config.chain.allow_missing_mechid,
    )

    metagraph = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
    burn_hotkey = chain.read_subnet_owner_hotkey(subtensor, netuid=config.chain.netuid)
    _require_burn_target(metagraph, burn_hotkey, netuid=config.chain.netuid)

    # --- 2. the day's data, and what the chain says about it -------------
    say(f"fetching the snapshot for {day.isoformat()}")
    snapshot = db.fetch_day(day)

    say(f"reading commitments for {len(snapshot.eligible_miners)} eligible miners")
    entries, notes = _chain_entries(
        snapshot, metagraph=metagraph, subtensor=subtensor, netuid=config.chain.netuid
    )

    say(f"verifying {len(entries)} commitments against Hugging Face")
    validations = tuple(registry.validate(entries))
    valid = [result for result in validations if result.valid]
    say(f"{len(valid)} of {len(validations)} models are eligible")

    # --- 3. ground truth, fixed before any model sees it -----------------
    say(
        f"labelling {len(snapshot.test_items)} test and "
        f"{len(snapshot.production_items)} production items"
    )
    labelled = label_items(
        _label_items_for(snapshot),
        client=labeller,
        policy=policy,
        batch_size=config.labelling.batch_size,
    )

    corpus = build_corpus(
        content_hash=snapshot.manifest.content_hash,
        test_items=_labelled_test(snapshot, labelled),
        production_items=_labelled_production(snapshot, labelled),
    )
    say(f"corpus is {len(corpus.items)} items")

    # --- 4. every eligible model over its own view of the corpus ---------
    evaluations: list[MinerEvaluation] = []
    for position, result in enumerate(valid, start=1):
        say(f"evaluating {position}/{len(valid)}: {result.hotkey} ({result.hf_repo})")
        target = MinerTarget(
            hotkey=result.hotkey,
            hf_repo=result.hf_repo,
            hf_revision=result.hf_revision,
        )
        # A download that fails is the miner's failure, not the cycle's: every
        # validator would hit it at once, so it is scored rather than raised.
        try:
            model_path = download_checkpoint(target)
        except EvaluationError as exc:
            evaluations.append(
                _unevaluable(
                    target, corpus.for_miner(result.hotkey), reason=str(exc), config=config
                )
            )
            continue

        evaluations.append(
            evaluate_miner(
                engine=build_engine(
                    model_path,
                    dtype=config.evaluation.torch_dtype,
                    device=config.evaluation.device,
                    revision=result.hf_revision,
                ),
                target=target,
                # Self-exclusion: a miner is never scored on content its own
                # users wrote. Done here rather than inside the runner so the
                # corpus stays one shared object.
                items=corpus.for_miner(result.hotkey),
                policy=policy,
                policy_version=policy_version,
                batch_size=config.evaluation.batch_size,
                deadline_seconds=config.evaluation.model_timeout_seconds,
            )
        )

    # --- 5. score both halves and reduce to one vector -------------------
    say("scoring")
    dataset_scores = score_dataset(_dataset_submissions(snapshot, labelled))
    ranking = score_models(
        [
            ModelEvaluation(
                hotkey=evaluation.hotkey,
                correct_count=evaluation.correct,
                scored_count=evaluation.total,
                total_tokens=evaluation.total_tokens,
                token_reading_count=evaluation.scored_token_items,
            )
            for evaluation in evaluations
        ],
        scoring=config.scoring,
    )
    split = combine_weights(
        dataset_scores=dataset_scores,
        model_scores=list(ranking.scores),
        scoring=config.scoring,
        burn_hotkey=burn_hotkey,
    )

    return CycleResult(
        day=day,
        snapshot=snapshot,
        labelled=labelled,
        corpus=corpus,
        validations=validations,
        evaluations=tuple(evaluations),
        ranking=ranking,
        split=split,
        burn_hotkey=burn_hotkey,
        metagraph=metagraph,
        started_at=now,
        notes=tuple(notes),
    )


def submit_weights(
    result: CycleResult, *, config: Config, subtensor: Any, wallet: Any, mechid: int | None
) -> str | None:
    """Send the cycle's vector to the chain.

    Resolved against a **fresh** metagraph, not the one the cycle scored
    against. A miner can deregister during a fourteen-hour cycle, and its UID is
    reassigned immediately, so submitting against a stale snapshot pays a
    stranger. Hotkeys that vanished are dropped and reported rather than
    aborting the whole submission over somebody else's exit.

    The submission gates are re-run here on freshly read hyperparameters. They
    are also checked at the top of the cycle, and both matter for opposite
    reasons: the early check refuses before an OpenAI bill is spent, and this
    one is the only check that reflects the chain as it is *now*. A cycle takes
    hours, and ``read_hyperparameters``' own docstring promised it ran
    "immediately before submission" while the only call site was step 1.
    Commit-reveal switched on mid-cycle is exactly the case the gate exists for,
    and exactly the case the early-only check missed.
    """
    return submit_allocation(
        weights=result.split.weights,
        burn_hotkey=result.burn_hotkey,
        config=config,
        subtensor=subtensor,
        wallet=wallet,
        mechid=mechid,
    )


def submit_allocation(
    *,
    weights: Mapping[str, int],
    burn_hotkey: str,
    config: Config,
    subtensor: Any,
    wallet: Any,
    mechid: int | None,
) -> str | None:
    """Resolve one hotkey->units allocation against the chain and send it.

    Split out of :func:`submit_weights` so a re-submission of an already
    computed day goes through the same gates and the same fresh-metagraph
    resolution. A validator has to re-post its vector between cycles — weights
    stop counting once ``activity_cutoff`` passes — and a second code path for
    that would be a second place for the burn target, the deregistration rule,
    or the commit-reveal gate to drift.
    """
    assert_submission_allowed(
        hyperparameters=chain.read_hyperparameters(subtensor, netuid=config.chain.netuid),
        capabilities=chain.detect_capabilities(subtensor),
        configured_version_key=config.chain.version_key,
        fail_on_weights_version_mismatch=config.chain.fail_on_weights_version_mismatch,
        allow_missing_mechid=config.chain.allow_missing_mechid,
    )

    fresh = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
    _require_burn_target(fresh, burn_hotkey, netuid=config.chain.netuid)
    registered, gone = fresh.partition_by_registration(list(weights))

    units = {hotkey: weights[hotkey] for hotkey in registered}
    if gone:
        # Their share is not redistributed. It goes to the burn hotkey, the
        # same place an unclaimed pool goes. Redistributing would let a
        # deregistration quietly raise everyone else's payout.
        units[burn_hotkey] = units.get(burn_hotkey, 0) + sum(weights[hotkey] for hotkey in gone)

    vector = to_u16_chain_vector(resolve_allocations(units, metagraph=fresh))
    return chain.submit_set_weights(
        subtensor,
        wallet=wallet,
        netuid=config.chain.netuid,
        vector=vector,
        version_key=config.chain.version_key,
        mechid=mechid,
    )


def build_results(result: CycleResult) -> tuple[MinerResult, ...]:
    """One published line per miner the cycle considered."""
    dataset_by_hotkey = {
        score.hotkey: score
        for score in score_dataset(_dataset_submissions(result.snapshot, result.labelled))
    }
    evaluation_by_hotkey = {item.hotkey: item for item in result.evaluations}
    score_by_hotkey = {score.hotkey: score for score in result.ranking.scores}
    rank_by_hotkey = {
        score.hotkey: position for position, score in enumerate(result.ranking.scores, start=1)
    }
    validation_by_hotkey = {item.hotkey: item for item in result.validations}

    rows: list[MinerResult] = []
    for miner in result.snapshot.eligible_miners:
        hotkey = miner.hotkey
        uid = result.metagraph.uid_for(hotkey)
        if uid is None:
            # Eligible per the DB layer but not on chain: it cannot hold a
            # weight, so it cannot hold a published result either.
            continue

        dataset = dataset_by_hotkey.get(hotkey)
        evaluation = evaluation_by_hotkey.get(hotkey)
        model_score = score_by_hotkey.get(hotkey)
        validation = validation_by_hotkey.get(hotkey)

        rows.append(
            MinerResult(
                hotkey=hotkey,
                uid=uid,
                weight=result.split.weights.get(hotkey, 0),
                dataset_submitted_count=dataset.submitted_count if dataset else 0,
                dataset_valid_count=dataset.valid_count if dataset else 0,
                dataset_score_micro=_micro(dataset.score) if dataset else 0,
                model_status=_model_status(validation, evaluation),
                model_items_scored=evaluation.total if evaluation else 0,
                model_correct_count=evaluation.correct if evaluation else 0,
                raw_accuracy_bp=evaluation.accuracy_bp if evaluation else 0,
                mean_total_tokens_milli=(
                    evaluation.mean_total_tokens_milli or 0 if evaluation else 0
                ),
                efficiency_penalty_bp=_penalty_bp(model_score, result.ranking),
                model_score_bp=_bp(model_score.score) if model_score else 0,
                model_rank=rank_by_hotkey.get(hotkey),
            )
        )
    return tuple(rows)


# ---------------------------------------------------------------------------
# Bridges between the modules' own vocabularies
# ---------------------------------------------------------------------------


def _unevaluable(
    target: MinerTarget, items: Any, *, reason: str, config: Config
) -> MinerEvaluation:
    """A model that could not be fetched, scored rather than skipped.

    Every validator would fail the same download at the same time, so raising
    here would take the subnet's submission down over one miner's repository.
    Scoring it wrong is the same treatment a model that loads and answers badly
    receives, which is the point: unavailable is a property of the submission.
    """
    from prometheon.evaluation.corpus import iter_batches
    from prometheon.evaluation.result import MinerEvaluationBuilder

    builder = MinerEvaluationBuilder(target.hotkey)
    for batch in iter_batches(items, config.evaluation.batch_size):
        builder.record_failed_batch(batch, reason=reason)
    return builder.build()


def _require_burn_target(metagraph: MetagraphView, burn_hotkey: str, *, netuid: int) -> None:
    """Refuse at minute zero if the burn target cannot hold a weight.

    Every unclaimed pool is routed to the subnet owner's hotkey, and
    :func:`resolve_allocations` is strict: a hotkey it cannot resolve to a uid
    raises rather than being dropped, because dropping one silently
    redistributes its emission to everybody else.

    A subnet owner's hotkey is **not** automatically registered as a neuron on
    its own subnet. When it is not, every cycle carrying any burn — which is
    most early cycles — reached submission and aborted there, after a full day
    of labelling and inference had already been paid for. The failure is the
    same either way; the cost of discovering it is not.

    Checked here rather than made survivable at submission because the
    alternatives are both worse: omitting the burn hands the chain's own
    normalisation the job of redistributing emission the design says nobody
    earned, and submitting nothing makes the validator look dead. A subnet whose
    owner holds no uid is misconfigured, and that is an operator's fix.
    """
    if metagraph.uid_for(burn_hotkey) is not None:
        return
    raise SubtensorError(
        f"the subnet owner hotkey {burn_hotkey!r} is not registered on netuid={netuid}, "
        f"so it cannot receive a weight. Every unclaimed pool burns to it, so this "
        f"cycle could compute a full result and then fail at submission. Register the "
        f"owner hotkey as a neuron on its own subnet before running a cycle"
    )


def _chain_entries(
    snapshot: DaySnapshot, *, metagraph: MetagraphView, subtensor: Any, netuid: int
) -> tuple[list[MinerEntry], list[str]]:
    """Read each eligible miner's commitment from the chain.

    The DB layer's commitment set is *not* used as the source of truth. It is
    used only to know who to ask about, because a commitment that decides
    emissions has to come from the authority that a miner signed it with. A DB
    layer that fabricated one would be contradicted here.

    The UID comes from the snapshot the cycle is scoring, and travels with the
    hotkey, because ``get_commitment`` is keyed by UID and a UID is a recycled
    slot.
    """
    entries: list[MinerEntry] = []
    notes: list[str] = []
    frozen: dict[str, FrozenCommitment] = {
        commitment.hotkey: commitment for commitment in snapshot.model_commitments
    }

    for miner in snapshot.eligible_miners:
        uid = metagraph.uid_for(miner.hotkey)
        if uid is None:
            notes.append(f"{miner.hotkey} is eligible but not registered on netuid={netuid}")
            continue
        try:
            commitment = read_commitment(subtensor, netuid=netuid, hotkey=miner.hotkey, uid=uid)
        except CommitmentError as exc:
            entries.append(MinerEntry(uid=uid, hotkey=miner.hotkey, decode_error=str(exc)))
            continue

        # The block settles duplicate claims on one revision SHA, and it has to
        # be read here or it is not read at all. Leaving it unset was the whole
        # of the defect: every entry carried the same default, the sort
        # collapsed to ascending uid, and a low-uid mirror beat the model's
        # author every time — the outcome `_duplicate_losers` names as the
        # strictly worse attack.
        #
        # A read failure is not fatal to the cycle. The miner keeps their entry
        # and takes `UNKNOWN_COMMIT_BLOCK`, which sorts last, so an unreadable
        # block costs its owner every tie instead of winning them.
        try:
            commit_block = read_commitment_block(subtensor, netuid=netuid, hotkey=miner.hotkey)
        except CommitmentError as exc:
            commit_block = UNKNOWN_COMMIT_BLOCK
            notes.append(
                f"{miner.hotkey}: commitment block unreadable ({exc}); it loses any "
                "duplicate-model tie it enters"
            )

        declared = frozen.get(miner.hotkey)
        if declared is not None and commitment is not None:
            # Note the field names differ: the DB layer's record calls it
            # `revision_sha`, the chain codec calls it `hf_revision`. Two
            # distinct types share the name `ModelCommitment`, which is why
            # they are imported under aliases at the top of this module.
            if (declared.hf_repo, declared.revision_sha) != (
                commitment.hf_repo,
                commitment.hf_revision,
            ):
                # Recorded, not acted on. The chain wins either way; the note
                # exists so a DB layer drifting from the chain is visible in the
                # published record rather than only in one validator's logs.
                notes.append(
                    f"{miner.hotkey}: the DB layer's frozen commitment disagrees with "
                    "the chain; the chain value is used"
                )

        chain_commitment: ChainCommitment | None = commitment
        entries.append(
            MinerEntry(
                uid=uid,
                hotkey=miner.hotkey,
                commitment=chain_commitment,
                commit_block=commit_block,
            )
        )
    return entries, notes


def _label_items_for(snapshot: DaySnapshot) -> list[LabelItem]:
    """Everything to be labelled, in one deterministic interleaved order.

    ``expected_violating`` is the prior, never the answer: test content was
    submitted *because* someone believed it violates, production content has a
    low base rate. Labelling uses it only to tell a legitimately unanimous batch
    apart from a subverted one.

    **The order is derived from the day's content hash**, using the same keyed
    digest :func:`~prometheon.evaluation.corpus.build_corpus` sorts by. Two
    properties follow, and the labelling path had neither.

    *Every validator batches identically.* Evaluation went to some trouble to
    make its order reproducible from the snapshot alone; labelling took whatever
    order the DB layer returned, so two validators could split the same corpus
    into different calls and disagree for reasons unrelated to the content.

    *Test and production interleave.* Concatenating the two put every miner's
    items in a contiguous run, so one successful injection inside a test batch
    landed almost entirely on the miner who authored that batch — it converted
    their own items to ``V = 100`` and poisoned ground truth at the same time.
    Scattering them means an injection that survives is diluted across the
    field rather than aimed, and it is what lets the base-rate tripwire see
    production items inside a batch that also carries test content. See
    ``_looks_subverted``: the two changes only work together.
    """
    content_hash = snapshot.manifest.content_hash
    items = [
        *(
            LabelItem(
                item_id=item.id, content=item.content, expected_violating=item.claimed_violating
            )
            for item in snapshot.test_items
        ),
        *(
            LabelItem(item_id=item.id, content=item.content, expected_violating=False)
            for item in snapshot.production_items
        ),
    ]
    items.sort(key=lambda item: (corpus_order_key(content_hash, item.item_id), item.item_id))
    return items


def _labelled_test(snapshot: DaySnapshot, labelled: LabelledCorpus) -> list[LabelledItem]:
    """Test items with whichever verdict the labeller gave them.

    Negatives included. An item a user submitted believing it broke the policy
    and that the labeller judged otherwise is a real negative example, and the
    hardest kind to get right. It still costs its author -- it counts in ``S``
    and not in ``V`` -- but the model half is measured on it rather than it
    being discarded after being paid for.
    """
    return [
        LabelledItem(
            item_id=item.id,
            content=item.content,
            violating=bool(labelled.labels.get(item.id)),
            source=ItemSource.TEST,
            contributor_hotkey=item.author_hotkey,
        )
        for item in snapshot.test_items
        if item.id in labelled.labels
    ]


def _labelled_production(snapshot: DaySnapshot, labelled: LabelledCorpus) -> list[LabelledItem]:
    """Production items with whichever verdict they received.

    This is where the negatives come from, so both labels are kept.
    """
    return [
        LabelledItem(
            item_id=item.id,
            content=item.content,
            violating=labelled.labels[item.id],
            source=ItemSource.PRODUCTION,
        )
        for item in snapshot.production_items
        if item.id in labelled.labels
    ]


def _dataset_submissions(
    snapshot: DaySnapshot, labelled: LabelledCorpus
) -> list[DatasetSubmission]:
    """Per-miner submitted and valid counts, derived by the validator itself.

    Counted from attributed test content and the validator's own labels rather
    than read from the DB layer, so the numerator and denominator of the dataset
    reward never come from the service being audited.

    Excluded items count as submitted but not valid: an item the labeller could
    not read is not a contribution, and treating it as neither would make
    unlabellable content free to submit.
    """
    submitted: dict[str, int] = {}
    valid: dict[str, int] = {}
    for miner in snapshot.eligible_miners:
        submitted.setdefault(miner.hotkey, 0)
        valid.setdefault(miner.hotkey, 0)

    for item in snapshot.test_items:
        if item.author_hotkey not in submitted:
            continue
        submitted[item.author_hotkey] += 1
        # Valid = correctly labelled: the labeller's independent verdict matches
        # the submitter's claim. Rewards accurate labelling of BOTH classes and
        # stays un-gameable — a member cannot inflate V by claiming everything
        # violates, because the labeller catches the false claim. An unlabellable
        # item (verdict None) matches neither claim and so counts submitted, not
        # valid.
        if labelled.labels.get(item.id) == item.claimed_violating:
            valid[item.author_hotkey] += 1

    return [
        DatasetSubmission(hotkey=hotkey, submitted_count=count, valid_count=valid.get(hotkey, 0))
        for hotkey, count in submitted.items()
    ]


def _model_status(
    validation: MinerValidation | None, evaluation: MinerEvaluation | None
) -> ModelStatus:
    if validation is None or (validation.reason is not None and not validation.hf_repo):
        return ModelStatus.NOT_COMMITTED
    if not validation.valid:
        return ModelStatus.INVALID
    if evaluation is None or not evaluation.has_token_reading:
        # Verified but served nothing: distinct from INVALID, because the fault
        # is at call time rather than in what was committed.
        return ModelStatus.UNREACHABLE
    return ModelStatus.EVALUATED


def _penalty_bp(model_score: Any, ranking: ModelRanking) -> int:
    """``lambda * t`` in basis points, or zero when the guard switched it off."""
    if model_score is None:
        return 0
    return int(Fraction(ranking.applied_lambda_bp) * model_score.efficiency_t)


def _bp(value: Fraction) -> int:
    return max(0, min(_BASIS_POINTS, int(value * _BASIS_POINTS)))


def _micro(value: Fraction) -> int:
    return max(0, int(value * 1_000_000))


__all__ = [
    "SCORING_VERSION",
    "CycleResult",
    "build_results",
    "run_cycle",
    "submit_weights",
]
