"""Largest-remainder allocation: the parts must always sum to the pool."""

from __future__ import annotations

from fractions import Fraction

import pytest

from prometheon.errors import ScoringError
from prometheon.scoring.allocation import AllocationTarget, allocate, to_micros

pytestmark = pytest.mark.unit


def _targets(**weights: int) -> list[AllocationTarget]:
    return [AllocationTarget(key=key, weight=Fraction(value)) for key, value in weights.items()]


def test_parts_sum_exactly_to_the_pool_when_shares_do_not_divide() -> None:
    parts = allocate(10, _targets(a=1, b=1, c=1))

    assert sum(parts.values()) == 10
    # 10/3 each: everyone floors to 3, and the stranded unit goes to the lowest
    # key because every remainder is identical.
    assert parts == {"a": 4, "b": 3, "c": 3}


def test_remainder_ties_do_not_depend_on_input_order() -> None:
    forward = allocate(10, _targets(a=1, b=1, c=1))
    reversed_order = allocate(10, _targets(c=1, b=1, a=1))

    assert forward == reversed_order


def test_largest_remainder_prefers_the_bigger_shortfall_over_the_lower_key() -> None:
    # Exact shares are 7/6 and 35/6: "b" is owed .833 of a unit against "a"'s
    # .166, so the stranded unit is b's even though "a" sorts first.
    parts = allocate(7, _targets(a=1, b=5))

    assert sum(parts.values()) == 7
    assert parts == {"a": 1, "b": 6}


def test_sums_exactly_across_many_awkward_pools() -> None:
    targets = _targets(a=7, b=11, c=13, d=17, e=19)

    for pool in (0, 1, 2, 3, 97, 1_000, 999_983, 1_000_000):
        parts = allocate(pool, targets)
        assert sum(parts.values()) == pool, pool
        assert all(value >= 0 for value in parts.values())


def test_a_zero_pool_allocates_zero_to_everyone() -> None:
    assert allocate(0, _targets(a=1, b=2)) == {"a": 0, "b": 0}


def test_keys_keep_the_order_they_were_given() -> None:
    parts = allocate(100, _targets(z=1, m=1, a=1))

    assert list(parts) == ["z", "m", "a"]


def test_weight_of_zero_earns_nothing_but_stays_in_the_map() -> None:
    parts = allocate(100, _targets(a=1, b=0))

    assert parts == {"a": 100, "b": 0}


def test_negative_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="must not be negative"):
        allocate(-1, _targets(a=1))


def test_negative_weight_is_refused() -> None:
    with pytest.raises(ScoringError, match="negative"):
        allocate(10, [AllocationTarget(key="a", weight=Fraction(-1))])


def test_empty_target_list_is_refused_rather_than_silently_returning_nothing() -> None:
    with pytest.raises(ScoringError, match="no targets"):
        allocate(10, [])


def test_all_zero_weights_is_refused_so_the_caller_must_decide_to_burn() -> None:
    with pytest.raises(ScoringError, match="all zero"):
        allocate(10, _targets(a=0, b=0))


def test_duplicate_keys_are_refused() -> None:
    with pytest.raises(ScoringError, match="duplicate"):
        allocate(
            10,
            [
                AllocationTarget(key="a", weight=Fraction(1)),
                AllocationTarget(key="a", weight=Fraction(1)),
            ],
        )


def test_to_micros_is_exact_for_representable_values() -> None:
    assert to_micros(Fraction(25)) == 25_000_000
    assert to_micros(Fraction(1, 2)) == 500_000
    assert to_micros(Fraction(0)) == 0


def test_to_micros_rounds_half_to_even_rather_than_drifting_upward() -> None:
    assert to_micros(Fraction(1, 3)) == 333_333
    assert to_micros(Fraction(1, 2_000_000)) == 0
    assert to_micros(Fraction(3, 2_000_000)) == 2
