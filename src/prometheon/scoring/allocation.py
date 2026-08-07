"""Largest-remainder allocation of an integer pool across weighted targets.

Weights are exact rationals, never floats. Two validators that reduce the same
inputs to a weight vector must produce the *same* vector; float arithmetic can
disagree in the last unit between machines, and a divergent weight vector costs
vtrust for no reason anyone can debug. :class:`~fractions.Fraction` makes the
arithmetic reproducible everywhere.

Largest remainder rather than plain rounding because the parts must sum to the
pool **exactly**. Rounding each share independently strands units, and stranded
units are emission that silently goes nowhere: neither to a miner nor to the
burn address, so nobody notices it is missing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from prometheon.errors import ScoringError

#: Signed payloads carry integers only, so a rational score is published as a
#: count of millionths. One part per million is finer than any decision this
#: subnet makes, and it round-trips identically for every reader.
MICROS_PER_UNIT: Final[int] = 1_000_000

#: Basis points denominator, used wherever a config ``*_bp`` field is applied.
BASIS_POINTS: Final[int] = 10_000


@dataclass(frozen=True, slots=True)
class AllocationTarget:
    """One claimant on the pool, with its relative (unnormalised) weight."""

    key: str
    weight: Fraction


def to_micros(value: Fraction) -> int:
    """An exact rational as an integer count of millionths.

    Banker's rounding on an exact rational, so the result depends only on the
    value, not on how it was computed or on the platform's float behaviour.
    """
    return round(value * MICROS_PER_UNIT)


def allocate(pool: int, targets: Sequence[AllocationTarget]) -> dict[str, int]:
    """Split ``pool`` integer units across ``targets`` in proportion to weight.

    The returned parts sum to ``pool`` exactly. Keys appear in the order they
    were given, so a caller that passes a ranked sequence gets a ranked map.

    Refuses to allocate when nothing is claimable: an empty target list, or
    weights that are all zero. Those are decisions about *whether* anyone is
    paid, which belong to the caller that knows what an empty field means
    (usually: the pool burns). Silently returning zeros here would let that
    decision be forgotten.
    """
    if pool < 0:
        raise ScoringError(f"weight pool must not be negative, got {pool}")
    if not targets:
        raise ScoringError("cannot allocate a pool across no targets")

    total = Fraction(0)
    seen: set[str] = set()
    for target in targets:
        if target.key in seen:
            raise ScoringError(f"duplicate allocation target: {target.key!r}")
        seen.add(target.key)
        if target.weight < 0:
            raise ScoringError(f"allocation weight for {target.key!r} is negative")
        total += target.weight

    if total == 0:
        raise ScoringError("cannot allocate a pool across targets whose weights are all zero")

    parts: dict[str, int] = {}
    remainders: list[tuple[Fraction, str]] = []
    for target in targets:
        exact = Fraction(pool) * target.weight / total
        whole = math.floor(exact)
        parts[target.key] = whole
        remainders.append((exact - whole, target.key))

    leftover = pool - sum(parts.values())
    assert 0 <= leftover < len(targets), "largest-remainder leftover is out of range"

    # Descending remainder, then key ascending: two targets whose remainders are
    # exactly equal must not be separated by iteration order.
    remainders.sort(key=lambda item: (-item[0], item[1]))
    for _, key in remainders[:leftover]:
        parts[key] += 1

    return parts


__all__ = [
    "BASIS_POINTS",
    "MICROS_PER_UNIT",
    "AllocationTarget",
    "allocate",
    "to_micros",
]
