"""Combining the two pools into one weight vector.

The pool is an integer count of weight units, and every share is a fixed slice
of the **total**, not a ratio of what happens to be claimed:

- ``dataset_share_bp`` (7000) goes to the top data contributors, split by score;
- ``model_rank_shares_bp`` (2500, 400, 100) go to the first, second and third
  model, placed as :mod:`prometheon.scoring.model` describes. A place nobody
  fills goes to the lowest-placed winner present, so two models take 25% and
  5%, and one model takes 30%.

A pool burns, to the subnet owner hotkey, only when nobody qualifies for it: no
data contributor scoring above zero, or no model scoring above zero. Anything a
config leaves out of the shares burns too, every cycle. The vector always sums
to the same total, so the burn line states exactly what was withheld instead of
leaving it to the chain's renormalisation.

A miner in both pools is one entry holding both amounts.
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
#: A million units is far finer than any distinction this scoring makes, and the
#: shipped shares divide it exactly.
DEFAULT_WEIGHT_POOL_UNITS: Final[int] = 1_000_000


@dataclass(frozen=True, slots=True)
class WeightSplit:
    """The finished vector, plus the parts it was built from.

    The parts are kept because "why did this hotkey get this weight" is the
    first question asked of any emission change, and reconstructing it from the
    total afterwards is guesswork.
    """

    total_units: int
    #: The vector: every earner, plus the burn hotkey carrying ``burn_units``.
    weights: Mapping[str, int]
    dataset_units: Mapping[str, int]
    model_units: Mapping[str, int]
    #: What each hotkey earned from the two pools, never including burn. The
    #: published record carries these, so a burn hotkey that also earned shows
    #: its earnings and its burn separately.
    earned_units: Mapping[str, int]
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
    """Reduce both pools to one ``{hotkey: weight_units}`` map.

    The result sums to ``total_units`` exactly, burn included.
    """
    if total_units <= 0:
        raise ScoringError(f"weight pool must be positive, got {total_units}")
    if not burn_hotkey:
        raise ScoringError("a burn hotkey is required; unallocated emission must have a target")

    dataset_units = allocate_dataset_pool(
        dataset_scores,
        pool=total_units * scoring.dataset_share_bp // BASIS_POINTS,
        top_n=scoring.dataset_top_n,
    )
    model_units = allocate_model_pool(
        model_scores,
        total_units=total_units,
        rank_shares_bp=scoring.model_rank_shares_bp,
        min_scored_items=scoring.model_min_scored_items,
    )

    earned: dict[str, int] = {}
    for source in (dataset_units, model_units):
        for hotkey, units in source.items():
            earned[hotkey] = earned.get(hotkey, 0) + units

    burn_units = total_units - sum(earned.values())
    if burn_units < 0:
        raise ScoringError(
            f"the shares allocated {total_units - burn_units} units from a pool of "
            f"{total_units}; they must not sum above 100%"
        )
    weights = dict(earned)
    if burn_units > 0:
        weights[burn_hotkey] = weights.get(burn_hotkey, 0) + burn_units

    return WeightSplit(
        total_units=total_units,
        weights=MappingProxyType(weights),
        dataset_units=MappingProxyType(dataset_units),
        model_units=MappingProxyType(model_units),
        earned_units=MappingProxyType(earned),
        burn_hotkey=burn_hotkey,
        burn_units=burn_units,
    )


__all__ = ["DEFAULT_WEIGHT_POOL_UNITS", "WeightSplit", "combine_weights"]
