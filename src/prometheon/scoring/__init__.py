"""Scoring: the pure reduction from a day's measurements to a weight vector.

Nothing in this package performs IO, reads a clock, or draws a random number.
Given the same measurements and the same :class:`~prometheon.config.ScoringConfig`
it returns the same integers on any machine, which is what lets a third party
recompute a validator's weights and disagree with evidence rather than opinion.

All arithmetic is exact rational (:class:`~fractions.Fraction`). There are no
floats anywhere in it: they cannot be canonicalised for signing, and they let
two honest validators reach different last units.
"""

from prometheon.scoring.allocation import (
    BASIS_POINTS,
    MICROS_PER_UNIT,
    AllocationTarget,
    allocate,
    to_micros,
)
from prometheon.scoring.combine import (
    DEFAULT_WEIGHT_POOL_UNITS,
    WeightSplit,
    combine_weights,
)
from prometheon.scoring.dataset import (
    DatasetScore,
    DatasetSubmission,
    allocate_dataset_pool,
    rank_dataset,
    score_dataset,
)
from prometheon.scoring.model import (
    ModelEvaluation,
    ModelRanking,
    ModelScore,
    allocate_model_pool,
    rank_models,
    score_models,
)

__all__ = [
    "BASIS_POINTS",
    "DEFAULT_WEIGHT_POOL_UNITS",
    "MICROS_PER_UNIT",
    "AllocationTarget",
    "DatasetScore",
    "DatasetSubmission",
    "ModelEvaluation",
    "ModelRanking",
    "ModelScore",
    "WeightSplit",
    "allocate",
    "allocate_dataset_pool",
    "allocate_model_pool",
    "combine_weights",
    "rank_dataset",
    "rank_models",
    "score_dataset",
    "score_models",
    "to_micros",
]
