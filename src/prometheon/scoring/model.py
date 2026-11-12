"""Model scoring: accuracy, less a size penalty, and a ranking that favours the first.

For every model that finished its corpus in time::

    raw_accuracy_bp = correct * 10,000 / scored, rounded half up
    size_penalty_bp = floor(100 * log2(P / 500,000,000)), 0 up to 500M parameters
    model_score_bp  = max(0, raw_accuracy_bp - size_penalty_bp)

Everything is an integer, and the record publishes all three, so anyone can
recompute a validator's ranking from its record.

**Ranking.** A model ranks only with a score above zero and at least
``model_min_scored_items`` scored items. Among the models whose score is
within :data:`RANK_BAND_BP` (one point) of the best, the one first committed
wins: the earliest ``commit_block``, which the epoch gives as the first block
at which that hotkey committed those weights, then the lower hotkey. It is
removed, and the same rule picks second, then third.

So an earlier model keeps its place unless a later one beats it by more than a
full point. A copy that is a hair better, or a fine-tune that moves a few
items, doesn't take the place of the model it was built from, and two
validators whose GPUs put a near-tie a hair apart still agree on the order.

A model that ran out of time, or scored nothing at all, gets no score.

This replaces v3.0's token-efficiency term, which measured the miner's
tokenizer rather than the model's cost (B29). Size is a cost every validator
pays the same way, and the parameter count is read the same way everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from prometheon.config import ScoringConfig
from prometheon.errors import ScoringError
from prometheon.scoring.allocation import BASIS_POINTS
from prometheon.scoring.size import size_penalty_bp

#: How far below the best score a model may sit and still win a place by
#: having been committed first: one point of accuracy.
RANK_BAND_BP: Final[int] = 100


def accuracy_bp(correct: int, scored: int) -> int:
    """``correct / scored`` in basis points, rounded half up; 0 when nothing was scored.

    The record publishes this as ``raw_accuracy_bp``, and the score is computed
    from the published value, so the two can never disagree.
    """
    if scored <= 0:
        return 0
    return (correct * BASIS_POINTS + scored // 2) // scored


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """What one miner's model did over the day's corpus."""

    hotkey: str
    correct_count: int
    scored_count: int
    #: The first block at which this hotkey committed these weights, from the
    #: signed epoch. Decides ranks within :data:`RANK_BAND_BP`.
    commit_block: int
    #: ``P``, read from the model's safetensors headers.
    parameters: int
    #: The model did not finish within the time limit, so it gets no score.
    timed_out: bool = False


@dataclass(frozen=True, slots=True)
class ModelScore:
    hotkey: str
    correct_count: int
    scored_count: int
    commit_block: int
    parameters: int
    raw_accuracy_bp: int
    size_penalty_bp: int
    #: ``max(0, raw_accuracy_bp - size_penalty_bp)``.
    score_bp: int


@dataclass(frozen=True, slots=True)
class ModelRanking:
    """Every model's score, and the order the ranked ones were placed in."""

    #: Every model that finished and scored at least one item, by hotkey.
    scores: tuple[ModelScore, ...]
    #: The models that rank, best first. Only the first places are paid.
    ranked: tuple[ModelScore, ...]

    def rank_of(self, hotkey: str) -> int | None:
        """A model's place, from 1, or ``None`` when it does not rank."""
        for position, score in enumerate(self.ranked, start=1):
            if score.hotkey == hotkey:
                return position
        return None


def score_models(evaluations: Iterable[ModelEvaluation], *, scoring: ScoringConfig) -> ModelRanking:
    """Score every model and rank the ones that qualify.

    A model that timed out, or whose corpus held no items, gets no score:
    accuracy is undefined for it, and inventing one would pay a rank share to a
    model that was never measured.
    """
    scores = [
        _score(evaluation)
        for evaluation in _validated(evaluations)
        if not evaluation.timed_out and evaluation.scored_count > 0
    ]
    scores.sort(key=lambda entry: entry.hotkey)
    return ModelRanking(
        scores=tuple(scores),
        ranked=rank_models(scores, min_scored_items=scoring.model_min_scored_items),
    )


def rank_models(scores: Iterable[ModelScore], *, min_scored_items: int) -> tuple[ModelScore, ...]:
    """Place every qualifying model: the first committed within a point of the best.

    Qualifying means a score above zero and at least ``min_scored_items``
    scored items. Each place goes to the model with the earliest
    ``commit_block``, then the lowest hotkey, among those scoring at least the
    best remaining score less :data:`RANK_BAND_BP`. Every key is part of a
    total order, so no place depends on the order the scores arrived in.

    ``min_scored_items`` is an eligibility floor, not a tie-break. Accuracy over
    a single item is 100% or 0% and carries no information, and a miner can
    steer its own corpus size, because its own test content is excluded from
    what it is scored on.
    """
    field: list[ModelScore] = []
    seen: set[str] = set()
    for entry in scores:
        if entry.hotkey in seen:
            raise ScoringError(f"duplicate model score for hotkey {entry.hotkey!r}")
        seen.add(entry.hotkey)
        if entry.score_bp > 0 and entry.scored_count >= min_scored_items:
            field.append(entry)

    ranked: list[ModelScore] = []
    while field:
        best = max(entry.score_bp for entry in field)
        contenders = [entry for entry in field if entry.score_bp >= best - RANK_BAND_BP]
        winner = min(contenders, key=lambda entry: (entry.commit_block, entry.hotkey))
        ranked.append(winner)
        field = [entry for entry in field if entry.hotkey != winner.hotkey]
    return tuple(ranked)


def allocate_model_pool(
    scores: Sequence[ModelScore],
    *,
    total_units: int,
    rank_shares_bp: Sequence[int],
    min_scored_items: int,
) -> dict[str, int]:
    """Pay the top places their shares of **total** emission.

    The shares are slices of the total, not ratios of a model pool: first place
    takes ``rank_shares_bp[0]`` basis points of ``total_units``, second the
    next, and so on. A place nobody fills goes to the lowest-placed winner
    present, so the whole model pool is paid whenever any model qualifies:
    with the shipped 25/4/1, two models take 25% and 5%, and one model takes
    all 30%. The pool burns only when no model qualifies.

    The places are worked out here from ``scores`` rather than trusted from the
    caller's order, because this function decides money. **A zero score is
    never paid**: :func:`rank_models` leaves it out, so a field where every
    model scored zero pays nobody and burns, instead of paying in hotkey order.

    Returns an empty map when no model earned anything.
    """
    if total_units < 0:
        raise ScoringError(f"weight pool must not be negative, got {total_units}")
    if not rank_shares_bp:
        raise ScoringError("model rank shares are empty; no model can be paid")
    if any(share < 0 for share in rank_shares_bp):
        raise ScoringError("model rank shares must not be negative")

    winners = rank_models(scores, min_scored_items=min_scored_items)[: len(rank_shares_bp)]
    if not winners:
        return {}
    shares = list(rank_shares_bp[: len(winners)])
    # Every place past the last winner rolls down to that winner.
    shares[-1] += sum(rank_shares_bp[len(winners) :])
    paid: dict[str, int] = {}
    for entry, share in zip(winners, shares, strict=True):
        units = total_units * share // BASIS_POINTS
        if units > 0:
            paid[entry.hotkey] = units
    return paid


def _score(evaluation: ModelEvaluation) -> ModelScore:
    raw = accuracy_bp(evaluation.correct_count, evaluation.scored_count)
    penalty = size_penalty_bp(evaluation.parameters)
    return ModelScore(
        hotkey=evaluation.hotkey,
        correct_count=evaluation.correct_count,
        scored_count=evaluation.scored_count,
        commit_block=evaluation.commit_block,
        parameters=evaluation.parameters,
        raw_accuracy_bp=raw,
        size_penalty_bp=penalty,
        score_bp=max(0, raw - penalty),
    )


def _validated(evaluations: Iterable[ModelEvaluation]) -> list[ModelEvaluation]:
    """Evaluations worth scoring, with malformed input refused outright."""
    checked: list[ModelEvaluation] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        if evaluation.hotkey in seen:
            raise ScoringError(f"duplicate model evaluation for hotkey {evaluation.hotkey!r}")
        seen.add(evaluation.hotkey)
        if evaluation.correct_count < 0 or evaluation.scored_count < 0:
            raise ScoringError(f"negative evaluation counts for {evaluation.hotkey!r}")
        if evaluation.correct_count > evaluation.scored_count:
            raise ScoringError(
                f"{evaluation.hotkey!r} has more correct items than scored "
                f"({evaluation.correct_count} > {evaluation.scored_count})"
            )
        if evaluation.commit_block < 0:
            raise ScoringError(f"{evaluation.hotkey!r} has a negative commit block")
        if evaluation.parameters < 0:
            raise ScoringError(f"{evaluation.hotkey!r} has a negative parameter count")
        checked.append(evaluation)
    return checked


__all__ = [
    "RANK_BAND_BP",
    "ModelEvaluation",
    "ModelRanking",
    "ModelScore",
    "accuracy_bp",
    "allocate_model_pool",
    "rank_models",
    "score_models",
]
