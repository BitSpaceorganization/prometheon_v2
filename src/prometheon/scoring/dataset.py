"""Data contribution scoring: amount times quality, balanced by construction.

For one miner's verified test items in the day, after near-duplicates have been
removed from the whole day's corpus (:mod:`prometheon.scoring.dedupe`)::

    S       = items left
    V_pos   = items claimed violating that the labeller confirmed violating
    V_neg   = items claimed compliant that the labeller confirmed compliant

    amount  = 2 * min(V_pos, V_neg)
    quality = (V_pos + V_neg) / S
    score   = amount * quality

``amount`` is what makes a flood worthless. The old ``V^2/S`` paid for any
confirmed claim, so thousands of obviously compliant posts, each confirmed,
outscored a careful contributor. Here a miner is paid for pairs: every
confirmed violating item needs a confirmed compliant one beside it, so flooding
one class adds nothing to ``amount`` and lowers ``quality`` with every item the
labeller rejects.

``quality`` keeps the old submission-accuracy idea: an item the labeller did
not confirm, or could not label at all, counts in ``S`` and not in either ``V``.

Everything is an exact :class:`~fractions.Fraction`. ``S == 0`` scores zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from prometheon.errors import ScoringError
from prometheon.scoring.allocation import AllocationTarget, allocate


@dataclass(frozen=True, slots=True)
class DatasetSubmission:
    """One miner's test content for the day, as this validator labelled it."""

    hotkey: str
    #: ``S``: the miner's items left after near-duplicate removal.
    submitted_count: int
    #: ``V_pos``: claimed violating, confirmed violating.
    violating_confirmed: int
    #: ``V_neg``: claimed compliant, confirmed compliant.
    compliant_confirmed: int


@dataclass(frozen=True, slots=True)
class DatasetScore:
    hotkey: str
    submitted_count: int
    violating_confirmed: int
    compliant_confirmed: int
    score: Fraction

    @property
    def valid_count(self) -> int:
        """Confirmed claims of either class, ``V_pos + V_neg``."""
        return self.violating_confirmed + self.compliant_confirmed


def score_dataset(submissions: Iterable[DatasetSubmission]) -> tuple[DatasetScore, ...]:
    """Score every submission and return them ranked, best first.

    Ties break on hotkey ascending. The rank decides who makes the paid cut, so
    it may not depend on the order rows arrived in.
    """
    scores: list[DatasetScore] = []
    seen: set[str] = set()
    for submission in submissions:
        if submission.hotkey in seen:
            raise ScoringError(f"duplicate dataset submission for hotkey {submission.hotkey!r}")
        seen.add(submission.hotkey)
        scores.append(
            DatasetScore(
                hotkey=submission.hotkey,
                submitted_count=submission.submitted_count,
                violating_confirmed=submission.violating_confirmed,
                compliant_confirmed=submission.compliant_confirmed,
                score=_score_one(submission),
            )
        )
    return rank_dataset(scores)


def rank_dataset(scores: Iterable[DatasetScore]) -> tuple[DatasetScore, ...]:
    """Order by score descending, hotkey ascending."""
    return tuple(sorted(scores, key=lambda entry: (-entry.score, entry.hotkey)))


def allocate_dataset_pool(
    scores: Sequence[DatasetScore], *, pool: int, top_n: int
) -> dict[str, int]:
    """Split the data pool across the top ``top_n`` scorers above zero.

    Proportional to score, so second place with half the score of first is paid
    half as much. Fewer than ``top_n`` scorers divide the whole pool among
    themselves.

    Returns an empty map when nobody scored above zero. The pool then burns,
    which :mod:`prometheon.scoring.combine` decides because it knows the burn
    target.
    """
    if top_n < 1:
        raise ScoringError(f"dataset_top_n must be at least 1, got {top_n}")
    if pool < 0:
        raise ScoringError(f"dataset pool must not be negative, got {pool}")

    # Re-ranked here rather than trusted: this function decides money, and the
    # caller's ordering is not part of its contract.
    earners = [entry for entry in rank_dataset(scores) if entry.score > 0][:top_n]
    if not earners or pool == 0:
        return {}

    return allocate(
        pool,
        [AllocationTarget(key=entry.hotkey, weight=entry.score) for entry in earners],
    )


def _score_one(submission: DatasetSubmission) -> Fraction:
    counts = (
        submission.submitted_count,
        submission.violating_confirmed,
        submission.compliant_confirmed,
    )
    if any(count < 0 for count in counts):
        raise ScoringError(
            f"negative dataset counts for {submission.hotkey!r}: S={counts[0]} "
            f"V_pos={counts[1]} V_neg={counts[2]}"
        )
    confirmed = submission.violating_confirmed + submission.compliant_confirmed
    if confirmed > submission.submitted_count:
        raise ScoringError(
            f"{submission.hotkey!r} has more confirmed items than submitted "
            f"({confirmed} > {submission.submitted_count}); the counts are inconsistent "
            "and must not be scored"
        )
    if submission.submitted_count == 0:
        return Fraction(0)
    amount = 2 * min(submission.violating_confirmed, submission.compliant_confirmed)
    quality = Fraction(confirmed, submission.submitted_count)
    return amount * quality


__all__ = [
    "DatasetScore",
    "DatasetSubmission",
    "allocate_dataset_pool",
    "rank_dataset",
    "score_dataset",
]
