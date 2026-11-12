"""The size rules: the 4B cap and floor(100 * log2(P / 500M)), exactly.

The expected penalties are worked by hand: log2(2) = 1, log2(3) = 1.58496...,
log2(4) = 2 and log2(8) = 3. The reference check below uses decimal arithmetic
at 60 digits, an independent route to the same number, never the code under
test.
"""

from __future__ import annotations

import decimal
import random

import pytest

from prometheon.errors import ScoringError
from prometheon.scoring.size import (
    MAX_MODEL_PARAMETERS,
    PENALTY_FREE_PARAMETERS,
    exceeds_size_cap,
    size_penalty_bp,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("parameters", "penalty"),
    [
        # 0.5B: no penalty at or under it, and the first parameter over it is
        # 100 * log2(1.000000002) = 0.0000003 bp, which floors to 0.
        (499_999_999, 0),
        (500_000_000, 0),
        (500_000_001, 0),
        # 1B: log2(2) = 1. One under is 99.9999997.
        (999_999_999, 99),
        (1_000_000_000, 100),
        (1_000_000_001, 100),
        # 1.5B: log2(3) = 1.58496, so 158.496 either side.
        (1_499_999_999, 158),
        (1_500_000_000, 158),
        (1_500_000_001, 158),
        # 2B: log2(4) = 2.
        (1_999_999_999, 199),
        (2_000_000_000, 200),
        (2_000_000_001, 200),
        # 4B: log2(8) = 3, the cap.
        (3_999_999_999, 299),
        (4_000_000_000, 300),
        (4_000_000_001, 300),
    ],
)
def test_the_penalty_at_each_size_the_team_named(parameters: int, penalty: int) -> None:
    assert size_penalty_bp(parameters) == penalty


def test_small_models_pay_nothing() -> None:
    assert size_penalty_bp(0) == 0
    assert size_penalty_bp(135_000_000) == 0
    assert size_penalty_bp(PENALTY_FREE_PARAMETERS) == 0


def test_the_penalty_is_an_integer() -> None:
    assert type(size_penalty_bp(1_234_567_891)) is int


def _reference(parameters: int) -> int:
    """floor(100 * log2(P / 500M)) in 60-digit decimals, for P over 500M."""
    with decimal.localcontext() as context:
        context.prec = 60
        ratio = decimal.Decimal(parameters) / decimal.Decimal(PENALTY_FREE_PARAMETERS)
        exact = 100 * ratio.ln() / decimal.Decimal(2).ln()
        return int(exact.to_integral_value(rounding=decimal.ROUND_FLOOR))


def test_the_exact_computation_agrees_with_an_independent_one_across_the_range() -> None:
    sizes = random.Random(20261112)  # noqa: S311 - reproducible sample, not a secret
    for _ in range(2000):
        parameters = sizes.randint(PENALTY_FREE_PARAMETERS + 1, MAX_MODEL_PARAMETERS)
        assert size_penalty_bp(parameters) == _reference(parameters), parameters


def test_every_step_lands_exactly_where_the_formula_puts_it() -> None:
    """The smallest P with each penalty from 1 to 300, from both sides.

    The boundary for penalty k is 500M * 2 ** (k / 100), rarely an integer, so
    this is where a float log2 would be most likely to round the wrong way.
    """
    for penalty in range(1, 301):
        with decimal.localcontext() as context:
            context.prec = 60
            boundary = decimal.Decimal(PENALTY_FREE_PARAMETERS) * (
                decimal.Decimal(2) ** (decimal.Decimal(penalty) / 100)
            )
        smallest = int(boundary.to_integral_value(rounding=decimal.ROUND_CEILING))
        assert size_penalty_bp(smallest) == penalty
        assert size_penalty_bp(smallest - 1) == penalty - 1


def test_the_penalty_never_falls_as_the_model_grows() -> None:
    previous = 0
    for parameters in range(PENALTY_FREE_PARAMETERS, MAX_MODEL_PARAMETERS + 1, 7_000_001):
        penalty = size_penalty_bp(parameters)
        assert penalty >= previous
        previous = penalty


@pytest.mark.parametrize("bad", [-1, True, 1.0e9])
def test_a_count_that_is_not_a_non_negative_integer_is_refused(bad: object) -> None:
    with pytest.raises(ScoringError):
        size_penalty_bp(bad)  # type: ignore[arg-type]


def test_the_cap_is_four_billion_and_inclusive() -> None:
    assert MAX_MODEL_PARAMETERS == 4_000_000_000
    assert not exceeds_size_cap(4_000_000_000)
    assert exceeds_size_cap(4_000_000_001)
