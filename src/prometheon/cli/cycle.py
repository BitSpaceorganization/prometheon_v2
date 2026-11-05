"""One validator cycle, end to end.

Separated from argument parsing so it can be driven by a test with fakes rather
than only by a terminal. Every external system arrives as a parameter; nothing
in here opens a connection or reads the environment.

The order of the stages is a contract:

1. **Chain gates first.** Commit-reveal, weights version, and mechid support are
   checked *before* a single API call is paid for. A cycle that cannot submit
   its result must not spend hours and a labelling bill discovering that.
2. **The signed epoch, then everything in it.** The hub's epoch fixes the day:
   who was registered at the frozen block, which models were frozen and paid
   for, which content counts. It is verified against a key pinned in config
   before anything in it is used, and its content files are hash-checked
   against it. Every validator therefore scores the same miners, the same
   models and the same content, whenever it runs (v2 read commitments live from
   the chain, so two validators could score different models: bug B6).
3. **Near-duplicates out, then label, then evaluate.** The day's corpus is
   reduced to the earliest copy of each text before anything is paid for.
   Ground truth is fixed before any model sees the corpus, so no model can
   influence what it is measured against.
4. **Score.** Weights and the published record are computed from the same
   in-memory result.

The cycle is a pure function of its inputs given the same external answers, and
:func:`run_cycle` returns everything it computed so a caller can print it, sign
it, or diff two validators' runs against each other.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from prometheon.chain import subtensor as chain
from prometheon.chain.commitment import ModelCommitment as ChainCommitment
from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import (
    assert_submission_allowed,
    resolve_allocations,
    to_u16_chain_vector,
)
from prometheon.config import Config
from prometheon.errors import (
    CommitmentError,
    ConfigError,
    SubtensorError,
    ValidatorFaultError,
)
from prometheon.evaluation.corpus import (
    EvaluationCorpus,
    ItemSource,
    LabelledItem,
    build_corpus,
    corpus_content_hash,
    corpus_order_key,
)
from prometheon.evaluation.preflight import check_evaluation_runtime
from prometheon.evaluation.result import MinerEvaluation
from prometheon.evaluation.runner import (
    CheckpointUnavailableError,
    MinerTarget,
    build_engine,
    discard_checkpoint,
    download_checkpoint,
    evaluate_miner,
)
from prometheon.hub.client import DayData, HubClient
from prometheon.hub.epoch import eligible_miners
from prometheon.hub.models import Epoch, EpochMiner, MinerResult, ModelStatus, Payment
from prometheon.labelling.batch import LabelledCorpus, label_items
from prometheon.labelling.client import ChatCompletionClient
from prometheon.labelling.prompt import LabelItem
from prometheon.registry.validation import (
    InvalidReason,
    MinerEntry,
    MinerValidation,
    ModelRegistry,
)
from prometheon.scoring.combine import WeightSplit, combine_weights
from prometheon.scoring.dataset import DatasetScore, DatasetSubmission, score_dataset
from prometheon.scoring.dedupe import CorpusEntry, drop_near_duplicates
from prometheon.scoring.model import ModelEvaluation, ModelRanking, score_models
from prometheon.version import SCORING_VERSION

_BASIS_POINTS = 10_000


@dataclass(frozen=True)
class Policy:
    """The content policy every verdict is judged against."""

    text: str
    #: The document's ``Version:`` line.
    version: str
    #: sha256 of the file's bytes. Must equal the epoch's ``policy.sha256``.
    sha256: str


@dataclass(frozen=True)
class CycleResult:
    """Everything one cycle computed. Nothing here has been sent anywhere."""

    day: dt.date
    data: DayData
    #: Item ids left after near-duplicate removal, across both files.
    kept_ids: frozenset[str]
    #: Each removed id, and the earlier kept id it duplicates.
    duplicates: Mapping[str, str]
    eligible: tuple[EpochMiner, ...]
    labelled: LabelledCorpus
    corpus: EvaluationCorpus
    #: Why each listed model was or was not evaluated, by hotkey.
    model_statuses: Mapping[str, ModelStatus]
    validations: tuple[MinerValidation, ...]
    evaluations: tuple[MinerEvaluation, ...]
    dataset_scores: tuple[DatasetScore, ...]
    ranking: ModelRanking
    split: WeightSplit
    burn_hotkey: str
    metagraph: MetagraphView
    started_at: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def epoch(self) -> Epoch:
        return self.data.epoch

    @property
    def epoch_content_hash(self) -> str:
        return self.data.verified.content_hash

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
    hub: HubClient,
    subtensor: Any,
    registry: ModelRegistry,
    labeller: ChatCompletionClient,
    policy: Policy,
    now: int,
    progress: Any = None,
    check_runtime: Callable[[], None] | None = None,
) -> CycleResult:
    """Run one day's cycle and return what it computed, without submitting.

    ``check_runtime`` raises :class:`ValidatorFaultError` when this validator
    cannot run a model at all. It defaults to a real canary on the configured
    device and dtype, and is checked before labelling and again after any model
    that failed every batch.
    """
    say = progress if progress is not None else (lambda _message: None)
    notes: list[str] = []
    if check_runtime is None:

        def check_runtime() -> None:
            check_evaluation_runtime(
                device=config.evaluation.device, dtype=config.evaluation.torch_dtype
            )

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

    # --- 2. the signed day ------------------------------------------------
    say(f"fetching the signed epoch for {day.isoformat()}")
    data = hub.fetch_day(day, scoring_version=SCORING_VERSION)
    epoch = data.epoch
    _require_matching_policy(epoch, policy)
    notes.extend(_feed_notes(data))

    floor = config.scoring.min_verified_active_members
    eligible = eligible_miners(epoch, min_verified_active_members=floor)
    eligible_by_hotkey = {miner.hotkey: miner for miner in eligible}
    notes.extend(_eligibility_notes(epoch, floor))
    say(f"{len(eligible)} of {len(epoch.miners)} registered miners are eligible")

    entries, statuses = _model_entries(epoch, eligible_by_hotkey)
    say(f"verifying {len(entries)} paid models of eligible miners against Hugging Face")
    validations = tuple(registry.validate(entries))
    _stop_on_upstream_outage(validations)
    valid = [result for result in validations if result.valid]
    say(f"{len(valid)} of {len(validations)} models can be evaluated")
    if valid:
        # Before anything is paid for: a validator that cannot run a model
        # would label the day and then score every miner zero (B10).
        say("checking that this validator can run a model")
        check_runtime()

    # --- 3. near-duplicates out, then ground truth -------------------------
    dedupe = drop_near_duplicates(
        _corpus_entries(data), max_hamming=config.scoring.near_duplicate_max_hamming
    )
    kept = frozenset(dedupe.kept)
    if dedupe.dropped:
        notes.append(f"{len(dedupe.dropped)} items removed as near-duplicates of earlier items")

    coldkeys = {miner.hotkey: miner.coldkey for miner in epoch.miners}
    to_label = _label_items_for(data, kept, coldkeys)
    say(f"labelling {len(to_label)} items")
    labelled = label_items(
        to_label,
        client=labeller,
        policy=policy.text,
        batch_size=config.labelling.batch_size,
    )

    corpus = build_corpus(
        content_hash=data.verified.content_hash,
        test_items=_labelled_test(data, kept, labelled, coldkeys),
        production_items=_labelled_production(data, kept, labelled),
    )
    say(f"corpus is {len(corpus.items)} items")

    # --- 4. every evaluable model over its own view of the corpus ---------
    evaluations: list[MinerEvaluation] = []
    for position, result in enumerate(valid, start=1):
        say(f"evaluating {position}/{len(valid)}: {result.hotkey} ({result.hf_repo})")
        target = MinerTarget(
            hotkey=result.hotkey, hf_repo=result.hf_repo, hf_revision=result.hf_revision
        )
        # Self-exclusion by coldkey: no model is scored on test content written
        # under its own coldkey, whichever hotkey carried it (B9).
        view = corpus.for_miner(result.hotkey, coldkeys[result.hotkey])
        try:
            model_path = download_checkpoint(target)
        except CheckpointUnavailableError as exc:
            # The repository's fault, so every validator sees it: score it.
            # Any other download failure is raised as ValidatorFaultError and
            # stops the cycle instead.
            evaluations.append(_unevaluable(target, view, reason=str(exc), config=config))
            notes.append(f"{result.hotkey}: {exc}")
            continue

        evaluation = evaluate_miner(
            engine=build_engine(
                model_path,
                dtype=config.evaluation.torch_dtype,
                device=config.evaluation.device,
                revision=result.hf_revision,
            ),
            target=target,
            items=view,
            policy=policy.text,
            policy_version=policy.version,
            batch_size=config.evaluation.batch_size,
            deadline_seconds=config.evaluation.model_timeout_seconds,
        )
        if evaluation.total and not evaluation.has_token_reading:
            # Every batch failed. Either the model is broken, which is the
            # miner's, or this validator stopped being able to run anything,
            # which must not be scored. The canary tells them apart.
            check_runtime()
            reasons = sorted(set(evaluation.failure_reasons))
            notes.append(f"{result.hotkey}: every batch failed ({'; '.join(reasons)[:300]})")
        evaluations.append(evaluation)
        if config.evaluation.discard_checkpoints:
            freed = discard_checkpoint(model_path)
            if freed:
                say(f"  discarded checkpoint, reclaimed {freed / 1024**3:.1f} GiB")

    # --- 5. score both pools and reduce to one vector ----------------------
    say("scoring")
    dataset_scores = score_dataset(_dataset_submissions(data, kept, labelled, eligible))
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
        data=data,
        kept_ids=kept,
        duplicates=dedupe.dropped,
        eligible=eligible,
        labelled=labelled,
        corpus=corpus,
        model_statuses=statuses,
        validations=validations,
        evaluations=tuple(evaluations),
        dataset_scores=dataset_scores,
        ranking=ranking,
        split=split,
        burn_hotkey=burn_hotkey,
        metagraph=metagraph,
        started_at=now,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class SubmittedAllocation:
    """What actually went on chain, which the published record has to describe."""

    receipt: str | None
    #: Units as sent, after deregistered hotkeys' units moved to burn.
    units: Mapping[str, int]
    #: Hotkeys that had deregistered by the time of sending.
    gone: tuple[str, ...]


def submit_weights(
    result: CycleResult, *, config: Config, subtensor: Any, wallet: Any, mechid: int | None
) -> SubmittedAllocation:
    """Send the cycle's vector to the chain, against a fresh metagraph."""
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
) -> SubmittedAllocation:
    """Resolve one hotkey->units allocation against the chain and send it.

    Resolved against a **fresh** metagraph, not the one the cycle started with.
    A miner can deregister during a long cycle, and its UID is reassigned
    immediately, so submitting against a stale snapshot pays a stranger.
    Hotkeys that vanished send their units to the burn hotkey rather than
    being redistributed: redistributing would let one deregistration quietly
    raise everyone else's payout.

    The submission gates are re-run here on freshly read hyperparameters.
    Commit-reveal switched on mid-cycle is exactly the case the gate exists
    for, and an early-only check would miss it.
    """
    assert_submission_allowed(
        hyperparameters=chain.read_hyperparameters(subtensor, netuid=config.chain.netuid),
        capabilities=chain.detect_capabilities(subtensor),
        configured_version_key=config.chain.version_key,
        fail_on_weights_version_mismatch=config.chain.fail_on_weights_version_mismatch,
        allow_missing_mechid=config.chain.allow_missing_mechid,
    )

    fresh = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
    registered, gone = fresh.partition_by_registration(list(weights))

    units = {hotkey: weights[hotkey] for hotkey in registered}
    if gone:
        units[burn_hotkey] = units.get(burn_hotkey, 0) + sum(weights[hotkey] for hotkey in gone)
    if units.get(burn_hotkey):
        # Only when something actually burns. v2 refused every re-post while
        # the owner was unregistered, even of a vector that burned nothing
        # (B19).
        _require_burn_target(fresh, burn_hotkey, netuid=config.chain.netuid)

    vector = to_u16_chain_vector(resolve_allocations(units, metagraph=fresh))
    receipt = chain.submit_set_weights(
        subtensor,
        wallet=wallet,
        netuid=config.chain.netuid,
        vector=vector,
        version_key=config.chain.version_key,
        mechid=mechid,
    )
    return SubmittedAllocation(receipt=receipt, units=dict(units), gone=tuple(gone))


def build_results(result: CycleResult) -> tuple[MinerResult, ...]:
    """One published line per eligible miner, and per hotkey with a listed model.

    ``weight`` is what the hotkey earned, never burn: the record carries burn
    separately, so its rows plus ``burned_weight`` are exactly the vector, even
    when the burn hotkey also earned (v2 folded burn into the owner's row, and
    a mirror then counted it twice: bug B5).
    """
    dataset_by_hotkey = {score.hotkey: score for score in result.dataset_scores}
    evaluation_by_hotkey = {item.hotkey: item for item in result.evaluations}
    score_by_hotkey = {score.hotkey: score for score in result.ranking.scores}
    rank_by_hotkey = {
        score.hotkey: position for position, score in enumerate(result.ranking.scores, start=1)
    }
    validation_by_hotkey = {item.hotkey: item for item in result.validations}
    listed = {model.hotkey for model in result.epoch.models}
    eligible = {miner.hotkey for miner in result.eligible}

    rows: list[MinerResult] = []
    for miner in result.epoch.miners:
        hotkey = miner.hotkey
        if hotkey not in eligible and hotkey not in listed:
            continue
        dataset = dataset_by_hotkey.get(hotkey)
        evaluation = evaluation_by_hotkey.get(hotkey)
        model_score = score_by_hotkey.get(hotkey)
        rows.append(
            MinerResult(
                hotkey=hotkey,
                uid=miner.uid,
                weight=result.split.earned_units.get(hotkey, 0),
                dataset_submitted_count=dataset.submitted_count if dataset else 0,
                dataset_valid_count=dataset.valid_count if dataset else 0,
                dataset_violating_confirmed=dataset.violating_confirmed if dataset else 0,
                dataset_compliant_confirmed=dataset.compliant_confirmed if dataset else 0,
                dataset_score_micro=_micro(dataset.score) if dataset else 0,
                model_status=_model_status(
                    hotkey,
                    listed=hotkey in listed,
                    statuses=result.model_statuses,
                    validation=validation_by_hotkey.get(hotkey),
                    evaluation=evaluation,
                ),
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


def _require_matching_policy(epoch: Epoch, policy: Policy) -> None:
    """The policy this validator labels against must be the one the day froze.

    Labels and model prompts both carry the policy text, so a validator on a
    different text computes different ground truth from identical content.
    v2 published only the version line and never compared the text.
    """
    if policy.sha256 != epoch.policy.sha256:
        raise ConfigError(
            f"content_policy.md hashes to {policy.sha256[:12]}, but the epoch for "
            f"{epoch.day.isoformat()} froze policy {epoch.policy.version!r} "
            f"({epoch.policy.sha256[:12]}). Run from the release that ships that policy; "
            "labelling against another text would score every miner on different ground truth"
        )


def _feed_notes(data: DayData) -> list[str]:
    notes: list[str] = []
    feed = data.feed
    if feed.rejected:
        reasons: dict[str, int] = {}
        for line in feed.rejected:
            reasons[f"{line.file}: {line.reason}"] = (
                reasons.get(f"{line.file}: {line.reason}", 0) + 1
            )
        summary = "; ".join(f"{count} x {reason}" for reason, count in sorted(reasons.items()))
        notes.append(f"{len(feed.rejected)} feed lines rejected ({summary})")
    if feed.hash_mismatches:
        notes.append(
            f"{feed.hash_mismatches} test lines carried a normalized_sha256 that differs from "
            "this validator's recomputation; the recomputed value was used"
        )
    return notes


def _eligibility_notes(epoch: Epoch, floor: int) -> list[str]:
    """Where the hub's flag and this validator's own floor disagree."""
    notes: list[str] = []
    for miner in epoch.miners:
        clears = miner.verified_active_members >= floor
        if miner.eligible != clears:
            flag = "eligible" if miner.eligible else "ineligible"
            notes.append(
                f"{miner.hotkey}: the hub marks it {flag} with "
                f"{miner.verified_active_members} verified active members against this "
                f"validator's floor of {floor}; both must agree, so it is not eligible here"
            )
    return notes


def _model_entries(
    epoch: Epoch, eligible: Mapping[str, EpochMiner]
) -> tuple[list[MinerEntry], dict[str, ModelStatus]]:
    """The frozen models this validator evaluates, and why the rest are not.

    Only ``paid`` models of eligible hotkeys are evaluated. The commitment is
    taken from the epoch, never from the chain: the frozen list is what every
    validator scores (B6).
    """
    entries: list[MinerEntry] = []
    statuses: dict[str, ModelStatus] = {}
    for model in epoch.models:
        if model.payment is Payment.UNPAID:
            statuses[model.hotkey] = ModelStatus.UNPAID
            continue
        if model.payment is Payment.DUPLICATE:
            statuses[model.hotkey] = ModelStatus.DUPLICATE
            continue
        miner = eligible.get(model.hotkey)
        if miner is None:
            statuses[model.hotkey] = ModelStatus.INELIGIBLE
            continue
        try:
            commitment: ChainCommitment | None = ChainCommitment(
                hf_repo=model.repo, hf_revision=model.revision
            )
            decode_error = None
        except CommitmentError as exc:
            commitment, decode_error = None, str(exc)
        entries.append(
            MinerEntry(
                uid=miner.uid,
                hotkey=model.hotkey,
                commitment=commitment,
                decode_error=decode_error,
                commit_block=model.commit_block,
                expected_fingerprint=model.fingerprint,
            )
        )
    return entries, statuses


def _stop_on_upstream_outage(validations: tuple[MinerValidation, ...]) -> None:
    """Hugging Face not answering this validator must not score a miner out.

    Checked before labelling, so stopping costs nothing but a retry. v2 marked
    such a miner ineligible for the day on this validator only, so its vector
    diverged from everyone else's for a reason that had nothing to do with the
    model (B21).
    """
    unreachable = [
        result for result in validations if result.reason is InvalidReason.UPSTREAM_UNAVAILABLE
    ]
    if unreachable:
        raise ValidatorFaultError(
            f"Hugging Face could not be reached to verify {len(unreachable)} model(s), first "
            f"{unreachable[0].hotkey}: {unreachable[0].detail}. Stopping before anything is "
            "labelled; the previous vector stays in place until a retry succeeds"
        )


def _corpus_entries(data: DayData) -> list[CorpusEntry]:
    feed = data.feed
    return [
        *(CorpusEntry(line.item_id, line.created_at_micros, line.body) for line in feed.test),
        *(CorpusEntry(line.item_id, line.created_at_micros, line.body) for line in feed.production),
    ]


def _label_items_for(
    data: DayData, kept: frozenset[str], coldkeys: Mapping[str, str]
) -> list[LabelItem]:
    """Everything to be labelled, in one deterministic interleaved order.

    ``expected_violating`` is the prior, never the answer: for test content it
    is the submitter's claim, for production content a low base rate. The
    labelling tripwire uses it to tell a legitimately unanimous batch from a
    subverted labeller, and needs to know whose claim each prior is. That
    ``author`` is the miner's **coldkey**: two hotkeys under one coldkey are one
    source, so one party cannot make a forged unanimity look like two (B22).

    The order comes from the epoch's content hash, so every validator batches
    identically, and test and production items interleave, so an injection
    that survives one batch is spread across the field rather than aimed.
    """
    content_hash = data.verified.content_hash
    items = [
        *(
            LabelItem(
                item_id=line.item_id,
                content=line.body,
                expected_violating=line.claimed_violating,
                author=coldkeys[line.miner_hotkey],
            )
            for line in data.feed.test
            if line.item_id in kept
        ),
        *(
            LabelItem(
                item_id=line.item_id, content=line.body, expected_violating=False, author=None
            )
            for line in data.feed.production
            if line.item_id in kept
        ),
    ]
    items.sort(key=lambda item: (corpus_order_key(content_hash, item.item_id), item.item_id))
    return items


def _labelled_test(
    data: DayData, kept: frozenset[str], labelled: LabelledCorpus, coldkeys: Mapping[str, str]
) -> list[LabelledItem]:
    """Test items with whichever verdict the labeller gave them.

    Both classes, whatever the claim. An item whose claim the labeller rejected
    is still a real example with a real label, and often a hard one. It still
    costs its author, because it counts in ``S`` and in neither ``V``, but the
    models are measured on it rather than it being discarded after being paid
    for.
    """
    return [
        LabelledItem(
            item_id=line.item_id,
            content=line.body,
            violating=bool(labelled.labels.get(line.item_id)),
            source=ItemSource.TEST,
            contributor_hotkey=line.miner_hotkey,
            contributor_coldkey=coldkeys[line.miner_hotkey],
        )
        for line in data.feed.test
        if line.item_id in kept and line.item_id in labelled.labels
    ]


def _labelled_production(
    data: DayData, kept: frozenset[str], labelled: LabelledCorpus
) -> list[LabelledItem]:
    return [
        LabelledItem(
            item_id=line.item_id,
            content=line.body,
            violating=labelled.labels[line.item_id],
            source=ItemSource.PRODUCTION,
        )
        for line in data.feed.production
        if line.item_id in kept and line.item_id in labelled.labels
    ]


def _dataset_submissions(
    data: DayData,
    kept: frozenset[str],
    labelled: LabelledCorpus,
    eligible: tuple[EpochMiner, ...],
) -> list[DatasetSubmission]:
    """Per eligible miner, ``S``, ``V_pos`` and ``V_neg``, from its kept test items.

    Counted from the validator's own labels, so the numbers the data reward is
    computed from never come from the service that delivered the content. A
    claim counts only when the labeller confirms it, and only towards its own
    class. An item the labeller disagreed with, or could not label at all,
    counts in ``S`` and in neither ``V``: treating an unlabellable item as
    neither would make unlabellable content free to submit.
    """
    submitted: dict[str, int] = {miner.hotkey: 0 for miner in eligible}
    violating: dict[str, int] = {}
    compliant: dict[str, int] = {}

    for line in data.feed.test:
        if line.item_id not in kept or line.miner_hotkey not in submitted:
            continue
        submitted[line.miner_hotkey] += 1
        verdict = labelled.labels.get(line.item_id)
        if verdict is None or verdict != line.claimed_violating:
            continue
        tally = violating if line.claimed_violating else compliant
        tally[line.miner_hotkey] = tally.get(line.miner_hotkey, 0) + 1

    return [
        DatasetSubmission(
            hotkey=hotkey,
            submitted_count=count,
            violating_confirmed=violating.get(hotkey, 0),
            compliant_confirmed=compliant.get(hotkey, 0),
        )
        for hotkey, count in submitted.items()
    ]


def _unevaluable(
    target: MinerTarget, items: Any, *, reason: str, config: Config
) -> MinerEvaluation:
    """A model that could not be fetched, scored rather than skipped."""
    from prometheon.evaluation.corpus import iter_batches
    from prometheon.evaluation.result import MinerEvaluationBuilder

    builder = MinerEvaluationBuilder(target.hotkey)
    for batch in iter_batches(items, config.evaluation.batch_size):
        builder.record_failed_batch(batch, reason=reason)
    return builder.build()


def _require_burn_target(metagraph: MetagraphView, burn_hotkey: str, *, netuid: int) -> None:
    """Refuse at minute zero if the burn target cannot hold a weight.

    Every unclaimed share is routed to the subnet owner's hotkey, and
    :func:`resolve_allocations` is strict: a hotkey it cannot resolve to a uid
    raises rather than being dropped, because dropping one silently
    redistributes its emission to everybody else.

    A subnet owner's hotkey is **not** automatically registered as a neuron on
    its own subnet. Under the v3 rules nearly every cycle burns something (an
    unfilled model rank is enough), so a cycle that ran with an unregistered
    owner would compute a full result and then fail at submission.
    """
    if metagraph.uid_for(burn_hotkey) is not None:
        return
    raise SubtensorError(
        f"the subnet owner hotkey {burn_hotkey!r} is not registered on netuid={netuid}, "
        f"so it cannot receive a weight. Every unclaimed share burns to it, so this "
        f"cycle could compute a full result and then fail at submission. Register the "
        f"owner hotkey as a neuron on its own subnet before running a cycle"
    )


def _model_status(
    hotkey: str,
    *,
    listed: bool,
    statuses: Mapping[str, ModelStatus],
    validation: MinerValidation | None,
    evaluation: MinerEvaluation | None,
) -> ModelStatus:
    if not listed:
        return ModelStatus.NOT_COMMITTED
    if hotkey in statuses:
        return statuses[hotkey]
    if validation is None or not validation.valid:
        return ModelStatus.INVALID
    if evaluation is None or not evaluation.has_token_reading:
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
    "Policy",
    "SubmittedAllocation",
    "build_results",
    "run_cycle",
    "submit_allocation",
    "submit_weights",
]
