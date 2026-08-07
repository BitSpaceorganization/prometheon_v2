"""Dataset contribution scoring: ``V^2 / S``.

A miner is paid for the *violating* test content its users submitted, damped by
the fraction of its submissions that were violating at all::

    score = V * (V / S)

Both factors are needed. Paying ``V`` alone rewards volume, so the cheapest
strategy becomes flooding the queue with anything; paying ``V / S`` alone
rewards a single lucky item. The product rewards a miner that brings a lot of
genuinely policy-violating content and nothing else, and it degrades smoothly
rather than at a threshold, so there is no cliff to game around.

``S == 0`` scores zero. No error is raised: a miner with an empty day simply
does not place. It must never reach a division either.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from prometheon.errors import ScoringError
from prometheon.scoring.allocation import AllocationTarget, allocate


@dataclass(frozen=True, slots=True)
class DatasetSubmission:
    """One miner's test-content tally for the day.

    ``valid_count`` is the number of submitted items the labeller judged to
    violate the policy, not the number accepted by the DB layer.
    """

    hotkey: str
    submitted_count: int
    valid_count: int


@dataclass(frozen=True, slots=True)
class DatasetScore:
    hotkey: str
    submitted_count: int
    valid_count: int
    score: Fraction


def score_dataset(submissions: Iterable[DatasetSubmission]) -> tuple[DatasetScore, ...]:
    """Score every submission and return them ranked, best first.

    Ties break on hotkey ascending. The rank decides who makes the paid cut, so
    it may not depend on the order the DB layer happened to return rows in.
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
                valid_count=submission.valid_count,
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
    """Split the dataset pool across the top ``top_n`` contributors.

    Proportional to score, so second place with half the score of first is paid
    half as much. Fewer than ``top_n`` scorers divide the pool among themselves
    rather than leaving the rest unpaid: a small field is not a reason to burn
    emission that miners earned.

    Returns an empty map when nobody scored above zero. The pool then burns.
    That decision belongs to :mod:`prometheon.scoring.combine`, the only place
    that knows the burn target.
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
    if submission.submitted_count < 0 or submission.valid_count < 0:
        raise ScoringError(
            f"negative dataset counts for {submission.hotkey!r}: "
            f"submitted={submission.submitted_count} valid={submission.valid_count}"
        )
    if submission.valid_count > submission.submitted_count:
        raise ScoringError(
            f"{submission.hotkey!r} has more valid items than submitted "
            f"({submission.valid_count} > {submission.submitted_count}); "
            "the snapshot is inconsistent and must not be scored"
        )
    if submission.submitted_count == 0:
        return Fraction(0)
    return Fraction(submission.valid_count * submission.valid_count, submission.submitted_count)


__all__ = [
    "DatasetScore",
    "DatasetSubmission",
    "allocate_dataset_pool",
    "rank_dataset",
    "score_dataset",
]
