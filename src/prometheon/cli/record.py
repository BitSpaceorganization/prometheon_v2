"""The published record, built so it always reproduces the vector that was sent.

A mirror rebuilds a provider's vector from its record: each row's ``weight``,
plus ``burned_weight`` on the burn hotkey. That only works if the record says
exactly what was sent. v2 broke this twice (B5): it folded burn into the owner's
row, so a mirror counted it twice, and it described the vector as computed,
not as sent, so a miner that deregistered before the send kept its units in the
record while the provider had sent them to burn.

So the record is drafted when the cycle finishes and finished after the send,
from the units actually sent. A hotkey that turned out to be deregistered gets a
zero row and its units join ``burned_weight``, exactly as
:func:`~prometheon.cli.cycle.submit_allocation` moved them. Where the vector had
to be shaped to the chain's weight limits, the rows carry the shaped amounts,
and a uid added only to meet the chain's minimum gets a row of its own.
"""

from __future__ import annotations

from collections.abc import Mapping

from prometheon.cli.cycle import CycleResult, build_results
from prometheon.errors import ScoringError
from prometheon.hub.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.version import SCORING_VERSION


def draft_record(
    result: CycleResult,
    *,
    netuid: int,
    validator_hotkey: str,
    policy_version: str,
    started_at: int,
    completed_at: int,
) -> EvaluationSubmission:
    """The unsigned record of the vector as computed."""
    rows = build_results(result)
    labelled_test = sum(1 for item in result.corpus.items if item.source.value == "test")
    return EvaluationSubmission(
        date=result.day,
        netuid=netuid,
        validator_hotkey=validator_hotkey,
        scoring_version=SCORING_VERSION,
        policy_version=policy_version,
        epoch_content_hash=result.epoch_content_hash,
        corpus_content_hash=result.corpus_hash,
        labelled_test_count=labelled_test,
        labelled_production_count=len(result.corpus.items) - labelled_test,
        results=rows,
        burned_weight=result.split.burn_units,
        started_at=started_at,
        completed_at=completed_at,
    )


def finish_record(
    draft: EvaluationSubmission,
    *,
    sent: Mapping[str, int],
    burn_hotkey: str,
    uids: Mapping[str, int],
) -> EvaluationSubmission:
    """The record of the vector exactly as sent.

    Each row's ``weight`` becomes what its hotkey was sent: zero for one that
    deregistered before the send, the shaped amount where the vector was bent
    to the chain's limits. A hotkey that was sent units but has no row, which
    only happens to a uid added to meet the chain's minimum number of weights,
    gets a row with nothing but its weight. The burn hotkey keeps whatever it
    earned in its own row, and the rest of its units are ``burned_weight``.
    Rows plus ``burned_weight`` then reproduce ``sent`` to the unit.
    """
    rows: list[MinerResult] = []
    for row in draft.results:
        if row.hotkey == burn_hotkey:
            weight = min(row.weight, sent.get(burn_hotkey, 0))
        else:
            weight = sent.get(row.hotkey, 0)
        rows.append(row if weight == row.weight else row.model_copy(update={"weight": weight}))
    listed = {row.hotkey for row in draft.results}
    for hotkey in sorted(sent):
        if hotkey in listed or hotkey == burn_hotkey or sent[hotkey] <= 0:
            continue
        if hotkey not in uids:
            raise ScoringError(f"{hotkey} was sent weight, but the uid it was sent at is unknown")
        rows.append(_weight_only_row(hotkey, uid=uids[hotkey], weight=sent[hotkey]))
    burned = sum(sent.values()) - sum(row.weight for row in rows)
    if burned < 0:
        raise ScoringError("the record's rows add up to more than the vector that was sent")
    return draft.model_copy(
        update={"results": tuple(rows), "burned_weight": burned, "signature": ""}
    )


def _weight_only_row(hotkey: str, *, uid: int, weight: int) -> MinerResult:
    """A row for a uid that held weight only to satisfy the chain's minimum."""
    return MinerResult(
        hotkey=hotkey,
        uid=uid,
        weight=weight,
        dataset_submitted_count=0,
        dataset_valid_count=0,
        dataset_violating_confirmed=0,
        dataset_compliant_confirmed=0,
        dataset_score_micro=0,
        model_status=ModelStatus.NOT_COMMITTED,
        model_items_scored=0,
        model_correct_count=0,
        raw_accuracy_bp=0,
        mean_total_tokens_milli=0,
        efficiency_penalty_bp=0,
        model_score_bp=0,
    )


__all__ = ["draft_record", "finish_record"]
