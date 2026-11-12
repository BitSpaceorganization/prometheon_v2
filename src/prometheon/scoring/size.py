"""The model size rules: a hard cap, and a penalty that grows with every doubling.

``P`` is a model's parameter count: the sum of the element counts of every
tensor in every ``*.safetensors`` file of the repository at the frozen
revision, read from the files' headers (:mod:`prometheon.registry.safetensors`).

- **The cap.** ``P <= 4,000,000,000``. A larger model is invalid, "too large",
  and is never evaluated. Validators run every eligible model on one 24 GB card,
  and the cap is what keeps that true. The hub lists only models under it.
- **The penalty.** ``penalty_bp = floor(100 * log2(P / 500,000,000))`` when
  ``P > 500,000,000``, else 0. One point of accuracy for every doubling past
  half a billion parameters: 1B costs 100 bp, 2B costs 200 bp, 4B costs 300 bp.
  A bigger model has to be that much more accurate to rank level with a smaller
  one.

These are published rules the hub applies too, not settings: a validator on a
different cap would evaluate models the hub never charged for, or the reverse.

**Exact integers, no floats.** ``math.log2`` would be one rounding step from a
different answer on some value close to a boundary, and every validator must
compute the same penalty. ``100 * log2(P / B)`` is ``log2((P / B) ** 100)``, so
the penalty is the largest ``k`` with ``2 ** k * B ** 100 <= P ** 100``: a
question about two integers of a few thousand bits, answered from their bit
lengths and one comparison.
"""

from __future__ import annotations

from typing import Final

from prometheon.errors import ScoringError

#: The most parameters a model may have and still be evaluated.
MAX_MODEL_PARAMETERS: Final[int] = 4_000_000_000

#: Up to this many parameters, a model carries no penalty.
PENALTY_FREE_PARAMETERS: Final[int] = 500_000_000

#: Basis points of accuracy each doubling past the penalty-free size costs.
PENALTY_BP_PER_DOUBLING: Final[int] = 100


def size_penalty_bp(parameters: int) -> int:
    """``floor(100 * log2(parameters / 500,000,000))``, or 0 at or below 500M.

    Computed on integers only, so every validator gets the same number.
    """
    if isinstance(parameters, bool) or not isinstance(parameters, int):
        raise ScoringError(f"a parameter count must be an integer, got {parameters!r}")
    if parameters < 0:
        raise ScoringError(f"a parameter count cannot be negative, got {parameters}")
    if parameters <= PENALTY_FREE_PARAMETERS:
        return 0
    # floor(log2(N / D)) for N > D: with a = bitlen(N) - 1 and b = bitlen(D) - 1,
    # N / D lies in (2 ** (a - b - 1), 2 ** (a - b + 1)), so the answer is a - b
    # or one less, and one comparison decides which.
    # Annotated: `int ** int` is typed as possibly float, for negative powers.
    numerator: int = parameters**PENALTY_BP_PER_DOUBLING
    denominator: int = PENALTY_FREE_PARAMETERS**PENALTY_BP_PER_DOUBLING
    doublings = numerator.bit_length() - denominator.bit_length()
    if denominator << doublings > numerator:
        doublings -= 1
    return doublings


def exceeds_size_cap(parameters: int) -> bool:
    """Whether a model is too large to be evaluated at all."""
    return parameters > MAX_MODEL_PARAMETERS


__all__ = [
    "MAX_MODEL_PARAMETERS",
    "PENALTY_BP_PER_DOUBLING",
    "PENALTY_FREE_PARAMETERS",
    "exceeds_size_cap",
    "size_penalty_bp",
]
