"""Combining the two halves into one weight vector.

The pool is an integer count of weight units, split by ``dataset_share_bp``.
Everything that is not allocated to a miner goes to the subnet owner hotkey as
burn, so the vector always sums to the same total no matter how thin the field
is. That makes a day's emission legible: the burn line is the subnet saying
"nobody earned this", explicitly, rather than a shortfall the chain quietly
renormalises away.

Four cases have to survive:

- no dataset-eligible miners: the dataset half burns;
- no model-eligible miners: the model half burns;
- neither: everything burns;
- a miner in both halves: its units add, and it is one entry in the vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from prometheon.config import ScoringConfig
from prometheon.errors import ScoringError
from prometheon.scoring.allocation import BASIS_POINTS
from prometheon.scoring.dataset import DatasetScore, allocate_dataset_pool
from prometheon.scoring.model import ModelScore, allocate_model_pool

#: Resolution of the weight vector before the chain's own u16 normalisation.
#: A million units is far finer than any distinction this scoring makes, so
#: rounding here never decides anything.
DEFAULT_WEIGHT_POOL_UNITS: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class WeightSplit:
    """The finished vector, plus the halves it was built from.

    The halves are kept because "why did this hotkey get this weight" is the
    first question asked of any emission change, and reconstructing it from the
    total afterwards is guesswork.
    """

    total_units: int
    weights: Mapping[str, int]
    dataset_units: Mapping[str, int]
    model_units: Mapping[str, int]
    burn_hotkey: str
    burn_units: int


def combine_weights(
    *,
    dataset_scores: Sequence[DatasetScore],
    model_scores: Sequence[ModelScore],
    scoring: ScoringConfig,
    burn_hotkey: str,
    total_units: int = DEFAULT_WEIGHT_POOL_UNITS,
) -> WeightSplit:
    """Reduce both halves to one ``{hotkey: weight_units}`` map.

    The result sums to ``total_units`` exactly, burn included.
    """
    if total_units <= 0:
        raise ScoringError(f"weight pool must be positive, got {total_units}")
    if not burn_hotkey:
        raise ScoringError("a burn hotkey is required; unallocated emission must have a target")

    dataset_pool = total_units * scoring.dataset_share_bp // BASIS_POINTS
    # The remainder rather than a second share calculation, so the two halves
    # cannot drift apart by a rounding unit as the share is retuned.
    model_pool = total_units - dataset_pool

    dataset_units = allocate_dataset_pool(
        dataset_scores, pool=dataset_pool, top_n=scoring.dataset_top_n
    )
    model_units = allocate_model_pool(
        model_scores, pool=model_pool, rank_shares_bp=scoring.model_rank_shares_bp
    )

    weights: dict[str, int] = {}
    for source in (dataset_units, model_units):
        for hotkey, units in source.items():
            weights[hotkey] = weights.get(hotkey, 0) + units

    burn_units = total_units - sum(weights.values())
    assert burn_units >= 0, "allocated more weight than the pool holds"
    if burn_units > 0:
        weights[burn_hotkey] = weights.get(burn_hotkey, 0) + burn_units

    return WeightSplit(
        total_units=total_units,
        weights=MappingProxyType(weights),
        dataset_units=MappingProxyType(dataset_units),
        model_units=MappingProxyType(model_units),
        burn_hotkey=burn_hotkey,
        burn_units=burn_units,
    )


__all__ = ["DEFAULT_WEIGHT_POOL_UNITS", "WeightSplit", "combine_weights"]
