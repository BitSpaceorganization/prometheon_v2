"""S5: a vector is bent to the subnet's weight limits before it is sent.

The chain refuses a weight vector naming fewer uids than
``min(subnet size, min_allowed_weights)``, or giving one uid a larger share than
``max_weight_limit / 65535``, unless it is a single self-weight. v2 read neither
limit, so a day on which everything burned could be refused at the last step.
Every expected number here is worked out by hand in the comments.
"""

from __future__ import annotations

import pytest

from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import (
    ChainHyperparameters,
    ChainWeightVector,
    ShapedAllocation,
    chain_refusal,
    resolve_allocations,
    sdk_emitted,
    shape_for_chain,
    to_u16_chain_vector,
)
from prometheon.errors import WeightSubmissionError

pytestmark = pytest.mark.unit

OWNER, VALIDATOR, A, B, C = "5Owner", "5Validator", "5MinerA", "5MinerB", "5MinerC"
VIEW = MetagraphView(
    netuid=1,
    uids=(0, 1, 2, 3, 4),
    hotkeys=(OWNER, VALIDATOR, A, B, C),
    stake_rao=(1, 1000, 1, 1, 1),
    validator_permit=(False, True, False, False, False),
)
DEFAULTS = ChainHyperparameters(weights_version=0, commit_reveal_enabled=False)
MIN_FOUR = ChainHyperparameters(
    weights_version=0, commit_reveal_enabled=False, min_allowed_weights=4
)
#: 32768 / 65535 = 50.0008%: one uid may hold just over half the vector.
HALF = ChainHyperparameters(weights_version=0, commit_reveal_enabled=False, max_weight_limit=32_768)


def _vector(units: dict[str, int]) -> ChainWeightVector:
    return to_u16_chain_vector(resolve_allocations(units, metagraph=VIEW))


def _refusal(
    units: dict[str, int], limits: ChainHyperparameters, validator: str = VALIDATOR
) -> str | None:
    return chain_refusal(
        _vector(units),
        hyperparameters=limits,
        subnet_n=len(VIEW),
        validator_uid=VIEW.uid_for(validator),
    )


def _shape(
    units: dict[str, int], limits: ChainHyperparameters, validator: str = VALIDATOR
) -> ShapedAllocation:
    return shape_for_chain(
        units,
        hyperparameters=limits,
        metagraph=VIEW,
        validator_hotkey=validator,
        burn_hotkey=OWNER,
    )


def test_on_the_default_limits_nothing_changes() -> None:
    units = {A: 600_000, B: 400_000}
    shaped = _shape(units, DEFAULTS)
    assert shaped.units == units
    assert (shaped.padded, shaped.capped) == ((), ())


class TestTooFewWeights:
    def test_a_burn_only_vector_is_refused_as_it_stands(self) -> None:
        assert _refusal({OWNER: 1_000_000}, MIN_FOUR) == (
            "1 weights, and the subnet requires at least 4"
        )

    def test_it_is_padded_with_the_lowest_uids_and_then_accepted(self) -> None:
        # min(5 neurons, 4) = 4 weights; the owner holds one, so three are added:
        # uids 2, 3, 4 (uid 1 is the validator itself and is skipped). Each gets
        # ceil(1,000,000 / 65535) = 16 units, the least that survives as a u16,
        # paid from the largest entry: 1,000,000 - 3 x 16 = 999,952.
        shaped = _shape({OWNER: 1_000_000}, MIN_FOUR)
        assert shaped.units == {OWNER: 999_952, A: 16, B: 16, C: 16}
        assert shaped.padded == (A, B, C)
        vector = _vector(shaped.units)
        assert sdk_emitted(vector) == ([0, 2, 3, 4], [65_535, 1, 1, 1])
        assert _refusal(shaped.units, MIN_FOUR) is None

    def test_the_minimum_is_capped_at_the_size_of_the_subnet(self) -> None:
        many = ChainHyperparameters(
            weights_version=0, commit_reveal_enabled=False, min_allowed_weights=64
        )
        # Five neurons, so five weights at most can be required; the validator
        # is used only because nothing else is left.
        shaped = _shape({OWNER: 1_000_000}, many)
        assert set(shaped.units) == {OWNER, VALIDATOR, A, B, C}
        assert _refusal(shaped.units, many) is None


class TestAShareOverTheCap:
    UNITS = {OWNER: 700_000, A: 250_000, B: 40_000, C: 10_000}  # noqa: RUF012

    def test_it_is_refused_as_it_stands(self) -> None:
        assert "over max_weight_limit=32768" in str(_refusal(self.UNITS, HALF))

    def test_a_miners_excess_is_burned(self) -> None:
        # A earned 60%; one uid may hold 499,992 units. The 100,008 A cannot be
        # paid is emission nobody qualifies for, so it burns to the owner.
        shaped = _shape({A: 600_000, B: 400_000}, HALF)
        assert shaped.units == {A: 499_992, B: 400_000, OWNER: 100_008}
        assert shaped.capped == (A,)
        assert _refusal(shaped.units, HALF) is None

    def test_an_excess_burn_is_shared_in_proportion(self) -> None:
        # cap = 1,000,000 x (32768 - 1) // 65535 = 499,992 (one u16 of margin).
        # The owner's excess, 700,000 - 499,992 = 200,008, goes to A, B and C in
        # proportion 25 : 4 : 1 of 30: 166,673.33, 26,667.73, 6,666.93. The two
        # units the floors leave go to the largest remainders, C then B.
        shaped = _shape(self.UNITS, HALF)
        assert shaped.units == {
            OWNER: 499_992,
            A: 250_000 + 166_673,
            B: 40_000 + 26_668,
            C: 10_000 + 6_667,
        }
        assert shaped.capped == (OWNER,)
        assert _refusal(shaped.units, HALF) is None

    def test_a_vector_too_short_for_the_cap_is_refused_rather_than_paid_out(self) -> None:
        # One entry can hold at most 49.9992%: the rest would have to go to uids
        # nobody scored. That is the owner's to fix, not the validator's to guess.
        with pytest.raises(WeightSubmissionError, match="raise max_weight_limit"):
            _shape({OWNER: 1_000_000}, HALF)


def test_a_single_self_weight_is_exempt_from_both_limits() -> None:
    both = ChainHyperparameters(
        weights_version=0, commit_reveal_enabled=False, min_allowed_weights=4, max_weight_limit=1
    )
    # The owner validating, with everything burned: one weight, on itself.
    shaped = _shape({OWNER: 1_000_000}, both, validator=OWNER)
    assert shaped.units == {OWNER: 1_000_000}
    assert _refusal(shaped.units, both, validator=OWNER) is None


def test_shaping_an_already_shaped_vector_changes_nothing() -> None:
    """A mirror re-shapes what its provider already shaped: it must land identically."""
    both = ChainHyperparameters(
        weights_version=0,
        commit_reveal_enabled=False,
        min_allowed_weights=4,
        max_weight_limit=32_768,
    )
    once = _shape({OWNER: 600_000, A: 400_000}, both).units
    twice = _shape(dict(once), both).units
    assert once == twice
    assert _refusal(once, both) is None
