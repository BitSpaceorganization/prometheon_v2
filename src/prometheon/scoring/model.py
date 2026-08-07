"""Model performance scoring, with an efficiency term that knows when to shut up.

Accuracy is the point: ``correct / scored``. The efficiency term exists only to
break the tie between two models that moderate equally well but cost the network
very different amounts of compute to run::

    t_i   = clamp(mean_tokens_i / P90(mean_tokens), 0, 1)
    score = accuracy * (1 - lambda * t_i)

Two guards keep it from doing damage:

- **lambda is capped in config**, so efficiency can never overturn an accuracy
  gap wider than that cap.
- **The spread guard.** If the coefficient of variation of mean token usage
  across the field is below ``efficiency_min_cv_bp``, every model costs about the
  same and the differences that remain are sampling noise: batching, a longer
  item drawn into one model's shuffle. Applying lambda there would reorder close
  rankings at random, which is indistinguishable from a coin flip deciding
  emission. Below the threshold the term switches off entirely.

The comparison is done on exact rationals. ``CV < c`` is equivalent to
``variance * 10000^2 < c_bp^2 * mean^2`` for a positive mean, which avoids a
square root and therefore avoids a float ever touching a ranking decision.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from prometheon.config import ScoringConfig
from prometheon.errors import ScoringError
from prometheon.scoring.allocation import BASIS_POINTS, AllocationTarget, allocate

#: Where the token-cost scale is read off. The 90th percentile rather than the
#: maximum: one pathological model must not compress everyone else's ``t`` to
#: near zero and neutralise the term for the whole field.
_PERCENTILE: Final[Fraction] = Fraction(9, 10)


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """What one miner's deployed model did over the day's corpus.

    ``token_reading_count`` is separate from ``scored_count`` because a batch
    that fails the wrapper contract is scored incorrect but reports no usage.
    Averaging its zero tokens in would make a broken model look cheap.
    """

    hotkey: str
    correct_count: int
    scored_count: int
    total_tokens: int = 0
    token_reading_count: int = 0


@dataclass(frozen=True, slots=True)
class ModelScore:
    hotkey: str
    correct_count: int
    scored_count: int
    accuracy: Fraction
    #: ``None`` when the model never reported usage, which is distinct from
    #: zero. Publishing a zero here would read as "free inference" and make the
    #: least reliable model in the field look like the most efficient one.
    mean_total_tokens: Fraction | None
    efficiency_t: Fraction
    score: Fraction


@dataclass(frozen=True, slots=True)
class ModelRanking:
    """Ranked scores plus the evidence for how the efficiency term was applied.

    ``applied_lambda_bp`` is the lambda that actually ran, which is zero
    whenever a guard fired. Publishing it means an operator can tell "efficiency
    changed nothing" apart from "efficiency was switched off", without rerunning
    the cycle.
    """

    scores: tuple[ModelScore, ...]
    applied_lambda_bp: int
    p90_mean_tokens: Fraction
    mean_token_cv_micros: int


def score_models(evaluations: Iterable[ModelEvaluation], *, scoring: ScoringConfig) -> ModelRanking:
    """Score and rank every evaluated model, best first.

    Models with no scored items are dropped: accuracy is undefined for them, and
    inventing a value would pay a rank share to a model that was never measured.
    """
    measured = _validated(evaluations, min_scored_items=scoring.model_min_scored_items)
    if not measured:
        return ModelRanking(
            scores=(), applied_lambda_bp=0, p90_mean_tokens=Fraction(0), mean_token_cv_micros=0
        )

    means = [_mean_tokens(evaluation) for evaluation in measured]

    # The field statistics are computed over models that actually reported
    # usage, and nothing else.
    #
    # This is a griefing fix. A model that never reported a reading used to
    # contribute a mean of *zero* to the distribution. Zero there means no
    # measurement at all, but it reads as the cheapest model in the field. One
    # such entry drags the P90 down, so honest models clamp to t=1 and eat the
    # full discount, and it inflates the spread so the CV guard stops firing,
    # which switches lambda back on across a field the guard had correctly
    # classified as noise. That is enough to swap rank 1 and rank 2 on a token
    # difference that means nothing: 1500bp of total subnet emission moved by a
    # miner who earns nothing and pays nothing. Register a hotkey, commit a
    # valid model, let the deployment return 500. That is the whole attack.
    readings = [mean for mean in means if mean is not None]
    p90 = _percentile_90(readings) if readings else Fraction(0)
    cv_micros = _coefficient_of_variation_micros(readings) if readings else 0
    applied_lambda_bp = _applied_lambda_bp(readings, p90, scoring) if readings else 0
    lambda_fraction = Fraction(applied_lambda_bp, BASIS_POINTS)

    scores: list[ModelScore] = []
    for evaluation, mean in zip(measured, means, strict=True):
        accuracy = Fraction(evaluation.correct_count, evaluation.scored_count)
        if mean is None:
            # No reading is the *worst* efficiency, never the best. A model that
            # reports no usage cannot be shown to be cheap, and treating the
            # absence as zero would hand the maximum efficiency bonus to a
            # deployment that answered nothing. It would also give every model
            # an incentive to omit its usage figures.
            efficiency_t = Fraction(1)
        elif p90 == 0:
            efficiency_t = Fraction(0)
        else:
            efficiency_t = _clamp_unit(mean / p90)
        scores.append(
            ModelScore(
                hotkey=evaluation.hotkey,
                correct_count=evaluation.correct_count,
                scored_count=evaluation.scored_count,
                accuracy=accuracy,
                mean_total_tokens=mean,
                efficiency_t=efficiency_t,
                score=accuracy * (1 - lambda_fraction * efficiency_t),
            )
        )

    return ModelRanking(
        scores=rank_models(scores),
        applied_lambda_bp=applied_lambda_bp,
        p90_mean_tokens=p90,
        mean_token_cv_micros=cv_micros,
    )


def rank_models(scores: Iterable[ModelScore]) -> tuple[ModelScore, ...]:
    """Score descending, then accuracy descending, then hotkey ascending.

    Three keys because the first two really do tie: with the efficiency term off
    two models with identical accuracy score identically, and the difference
    between rank 1 and rank 2 is 1500bp of subnet emission. That may not be
    decided by whatever order a dict happened to iterate in.
    """
    return tuple(sorted(scores, key=lambda entry: (-entry.score, -entry.accuracy, entry.hotkey)))


def allocate_model_pool(
    scores: Sequence[ModelScore], *, pool: int, rank_shares_bp: Sequence[int]
) -> dict[str, int]:
    """Split the model pool across the ranked models by rank share.

    The shares are expressed against *total* miner emission (3000/1500/500bp of
    it), and renormalise across the models actually present: with two models
    they become 3000/4500 and 1500/4500 of this pool. A thin field pays its
    models more rather than burning the difference.

    **A zero score is never paid a rank share.** Without that filter, a model
    that answered nothing all day still collects. A miner whose chute returns
    HTTP 500 to every batch scores exactly zero, and with no honest competitor
    above it, took the entire model half; with two honest models present it
    still took 500bp of total miner emission off them for doing nothing. Worse,
    when the whole field scores zero (the validator's egress to Chutes is down,
    or every deployment is broken), ranking falls through to the
    hotkey-ascending tie-break, so half of miner emission would be distributed
    in lexicographic hotkey order, identically on every validator, which means
    chain consensus cannot correct it either.

    Returns an empty map when no model earned anything, leaving the burn to
    :mod:`prometheon.scoring.combine`. That is what ``docs/scoring.md``
    documents and what the dataset half already did.
    """
    if pool < 0:
        raise ScoringError(f"model pool must not be negative, got {pool}")
    if not rank_shares_bp:
        raise ScoringError("model rank shares are empty; no model can be paid")
    if any(share < 0 for share in rank_shares_bp):
        raise ScoringError("model rank shares must not be negative")

    earned = [entry for entry in rank_models(scores) if entry.score > 0]
    paid = earned[: len(rank_shares_bp)]
    shares = list(rank_shares_bp[: len(paid)])
    if not paid or pool == 0 or sum(shares) == 0:
        return {}

    return allocate(
        pool,
        [
            AllocationTarget(key=entry.hotkey, weight=Fraction(share))
            for entry, share in zip(paid, shares, strict=True)
        ],
    )


def _validated(
    evaluations: Iterable[ModelEvaluation], *, min_scored_items: int
) -> list[ModelEvaluation]:
    """Evaluations worth scoring, with malformed input rejected outright.

    ``min_scored_items`` is an eligibility floor, not a tie-break. Accuracy over
    a single item is 1 or 0 and carries no information, yet it sorts above a
    model that answered 4999 of 5000 correctly and takes twice its emission.
    A miner can steer its own corpus size too, because its own test content is
    excluded from what it is scored on, so a miner supplying most of a day's
    test content is measured on very little. Dropping the under-measured model
    makes the rank shares renormalise across those that were actually measured.
    """
    measured: list[ModelEvaluation] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        if evaluation.hotkey in seen:
            raise ScoringError(f"duplicate model evaluation for hotkey {evaluation.hotkey!r}")
        seen.add(evaluation.hotkey)
        if (
            evaluation.correct_count < 0
            or evaluation.scored_count < 0
            or evaluation.total_tokens < 0
            or evaluation.token_reading_count < 0
        ):
            raise ScoringError(f"negative evaluation counts for {evaluation.hotkey!r}")
        if evaluation.correct_count > evaluation.scored_count:
            raise ScoringError(
                f"{evaluation.hotkey!r} has more correct items than scored "
                f"({evaluation.correct_count} > {evaluation.scored_count})"
            )
        if evaluation.scored_count >= min_scored_items:
            measured.append(evaluation)
    return measured


def _mean_tokens(evaluation: ModelEvaluation) -> Fraction | None:
    """Mean tokens per item, or ``None`` when nothing was ever measured.

    ``None`` rather than zero. "No reading" and "zero tokens" are different
    facts, and collapsing them makes a model that answered nothing the cheapest
    in the field.
    """
    if evaluation.token_reading_count == 0:
        return None
    return Fraction(evaluation.total_tokens, evaluation.token_reading_count)


def _percentile_90(values: Sequence[Fraction]) -> Fraction:
    """Linear-interpolated 90th percentile over exact rationals."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = _PERCENTILE * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _coefficient_of_variation_micros(values: Sequence[Fraction]) -> int:
    """CV as millionths, for the record only. Never for the guard comparison.

    Reported rather than compared because a standard deviation is irrational and
    this is a lossy view of it. The guard below compares squares instead.
    """
    mean = _mean(values)
    if mean <= 0:
        return 0
    variance = _variance(values, mean)
    # isqrt over a scaled integer keeps this exact enough to report and free of
    # any float, at the cost of truncating the last millionth.
    scaled = variance * (1_000_000**2) / (mean * mean)
    return math.isqrt(scaled.numerator // scaled.denominator)


def _applied_lambda_bp(means: Sequence[Fraction], p90: Fraction, scoring: ScoringConfig) -> int:
    if p90 <= 0:
        # No token scale at all: every reading is zero, so there is nothing to
        # be efficient relative to.
        return 0
    mean = _mean(means)
    if mean <= 0:
        return 0
    variance = _variance(means, mean)
    threshold = Fraction(scoring.efficiency_min_cv_bp)
    # CV < threshold/10000  <=>  variance * 10000^2 < threshold^2 * mean^2.
    if variance * (BASIS_POINTS**2) < threshold * threshold * mean * mean:
        return 0
    return scoring.efficiency_lambda_bp


def _mean(values: Sequence[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def _variance(values: Sequence[Fraction], mean: Fraction) -> Fraction:
    """Population variance: this is the whole field of models, not a sample."""
    return sum(((value - mean) * (value - mean) for value in values), Fraction(0)) / len(values)


def _clamp_unit(value: Fraction) -> Fraction:
    if value < 0:
        return Fraction(0)
    if value > 1:
        return Fraction(1)
    return value


__all__ = [
    "ModelEvaluation",
    "ModelRanking",
    "ModelScore",
    "allocate_model_pool",
    "rank_models",
    "score_models",
]
