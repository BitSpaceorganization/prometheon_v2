"""The published record, built so it always reproduces the vector that was sent.

A mirror rebuilds a provider's vector from its record: each row's ``weight``,
plus ``burned_weight`` on the burn hotkey. That only works if the record says
exactly what was sent. v2 broke this twice (B5): it folded burn into the owner's
row, so a mirror counted it twice, and it described the vector as computed,
not as sent, so a miner that deregistered before the send kept its units in the
record while the provider had sent them to burn.

So the record is drafted when the cycle finishes and finished after the send:
any hotkey that turned out to be deregistered gets a zero row and its units
join ``burned_weight``, exactly as :func:`~prometheon.cli.cycle.submit_allocation`
moved them.
"""

from __future__ import annotations

from collections.abc import Collection

from prometheon.cli.cycle import CycleResult, build_results
from prometheon.errors import ScoringError
from prometheon.hub.models import EvaluationSubmission
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


def finish_record(draft: EvaluationSubmission, *, gone: Collection[str]) -> EvaluationSubmission:
    """The record of the vector as sent: deregistered hotkeys' units moved to burn."""
    total = sum(row.weight for row in draft.results) + draft.burned_weight
    rows = tuple(
        row.model_copy(update={"weight": 0}) if row.hotkey in gone else row for row in draft.results
    )
    burned = total - sum(row.weight for row in rows)
    if burned < 0:
        raise ScoringError("the record's rows add up to more than its vector")
    return draft.model_copy(update={"results": rows, "burned_weight": burned, "signature": ""})


__all__ = ["draft_record", "finish_record"]
