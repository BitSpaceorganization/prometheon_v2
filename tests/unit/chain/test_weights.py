"""Behaviour of the submission gates and the u16 conversion."""

from __future__ import annotations

import pytest

from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import (
    CHAIN_WEIGHT_UNITS,
    ChainCapabilities,
    ChainHyperparameters,
    ChainWeightVector,
    WeightAllocation,
    assert_submission_allowed,
    resolve_allocations,
    to_u16_chain_vector,
)
from prometheon.errors import (
    CommitRevealEnabledError,
    DuplicateWeightTargetError,
    WeightSubmissionError,
    WeightsVersionMismatchError,
)

pytestmark = pytest.mark.unit


def allocations(*pairs: tuple[str, int, int]) -> list[WeightAllocation]:
    return [WeightAllocation(hotkey=h, uid=u, units=n) for h, u, n in pairs]


class TestPolicyGates:
    def test_clean_state_passes(self) -> None:
        assert_submission_allowed(
            hyperparameters=ChainHyperparameters(weights_version=7, commit_reveal_enabled=False),
            capabilities=ChainCapabilities(sdk_version="10.5.0", supports_mechid=True),
            configured_version_key=7,
        )

    def test_commit_reveal_stops_submission(self) -> None:
        with pytest.raises(CommitRevealEnabledError) as excinfo:
            assert_submission_allowed(
                hyperparameters=ChainHyperparameters(weights_version=7, commit_reveal_enabled=True),
                capabilities=ChainCapabilities(sdk_version="10.5.0", supports_mechid=True),
                configured_version_key=7,
            )
        assert excinfo.value.code == "chain.commit_reveal_enabled"

    def test_commit_reveal_outranks_every_other_problem(self) -> None:
        """It is the one condition no flag can waive, so it is reported first."""
        with pytest.raises(CommitRevealEnabledError):
            assert_submission_allowed(
                hyperparameters=ChainHyperparameters(weights_version=9, commit_reveal_enabled=True),
                capabilities=ChainCapabilities(sdk_version="9.0.0", supports_mechid=False),
                configured_version_key=7,
            )

    def test_missing_mechid_fails_closed_by_default(self) -> None:
        with pytest.raises(WeightSubmissionError, match="mechid") as excinfo:
            assert_submission_allowed(
                hyperparameters=ChainHyperparameters(
                    weights_version=7, commit_reveal_enabled=False
                ),
                capabilities=ChainCapabilities(sdk_version="9.0.0", supports_mechid=False),
                configured_version_key=7,
            )
        # Its own code, not the base class's: the remediation for this gate is
        # "upgrade the SDK", which is unlike every other weight-submission
        # failure, and an operator routes on the code.
        assert excinfo.value.code == "chain.mechid_missing"

    def test_missing_mechid_can_be_waived_for_localnet(self) -> None:
        assert_submission_allowed(
            hyperparameters=ChainHyperparameters(weights_version=7, commit_reveal_enabled=False),
            capabilities=ChainCapabilities(sdk_version="9.0.0", supports_mechid=False),
            configured_version_key=7,
            allow_missing_mechid=True,
        )

    def test_version_mismatch_fails_closed_by_default(self) -> None:
        with pytest.raises(WeightsVersionMismatchError) as excinfo:
            assert_submission_allowed(
                hyperparameters=ChainHyperparameters(
                    weights_version=9, commit_reveal_enabled=False
                ),
                capabilities=ChainCapabilities(sdk_version="10.5.0", supports_mechid=True),
                configured_version_key=7,
            )
        assert excinfo.value.code == "chain.weights_version_mismatch"

    def test_version_mismatch_can_be_waived(self) -> None:
        assert_submission_allowed(
            hyperparameters=ChainHyperparameters(weights_version=9, commit_reveal_enabled=False),
            capabilities=ChainCapabilities(sdk_version="10.5.0", supports_mechid=True),
            configured_version_key=7,
            fail_on_weights_version_mismatch=False,
        )


class TestResolveAllocations:
    def snapshot(self) -> MetagraphView:
        return MetagraphView(
            netuid=42,
            block=100,
            uids=(7, 3, 91),
            hotkeys=("5Alice", "5Bob", "5Carol"),
            stake_rao=(0, 0, 0),
            validator_permit=(False, False, False),
        )

    def test_orders_by_uid_regardless_of_input_order(self) -> None:
        """Two validators that agree on the numbers must emit the same vector."""
        forward = resolve_allocations(
            {"5Alice": 10, "5Bob": 20, "5Carol": 30}, metagraph=self.snapshot()
        )
        reversed_input = resolve_allocations(
            {"5Carol": 30, "5Bob": 20, "5Alice": 10}, metagraph=self.snapshot()
        )
        assert forward == reversed_input
        assert [a.uid for a in forward] == [3, 7, 91]

    def test_an_unregistered_hotkey_raises_rather_than_vanishing(self) -> None:
        """Dropping it would silently hand its emission to everybody else."""
        with pytest.raises(WeightSubmissionError, match="not registered"):
            resolve_allocations({"5Mallory": 10}, metagraph=self.snapshot())

    def test_negative_units_are_rejected(self) -> None:
        with pytest.raises(WeightSubmissionError, match="non-negative"):
            resolve_allocations({"5Alice": -1}, metagraph=self.snapshot())

    def test_empty_input_resolves_to_nothing(self) -> None:
        assert resolve_allocations({}, metagraph=self.snapshot()) == []


class TestU16Conversion:
    @pytest.mark.parametrize(
        "units",
        [
            [1],
            [1, 1],
            [1, 1, 1],
            [1, 2, 3, 4, 5, 6, 7],
            [5000, 5000, 5000, 5000, 5000, 5000, 5000],  # seven equal thirds-ish
            [1] * 64,
            [1] * 256,
            [999_999_999, 1],
            [3000, 1500, 500],
            [10_000, 0, 0, 0],
            [7, 11, 13, 17, 19, 23, 29, 31, 37, 41],
            [500_000_000] * 3,
        ],
    )
    def test_the_vector_always_sums_to_the_chain_total(self, units: list[int]) -> None:
        vector = to_u16_chain_vector(
            allocations(*[(f"5hk{i:03d}", i, n) for i, n in enumerate(units)])
        )
        assert vector.total() == CHAIN_WEIGHT_UNITS
        assert all(0 <= weight <= CHAIN_WEIGHT_UNITS for weight in vector.weights)
        assert len(vector.weights) == len(units)

    def test_a_single_target_takes_everything(self) -> None:
        vector = to_u16_chain_vector(allocations(("5Alice", 7, 12345)))
        assert vector == ChainWeightVector(
            uids=(7,), weights=(CHAIN_WEIGHT_UNITS,), hotkeys=("5Alice",)
        )

    def test_proportions_are_preserved(self) -> None:
        vector = to_u16_chain_vector(
            allocations(("5Alice", 0, 3000), ("5Bob", 1, 1500), ("5Carol", 2, 500))
        )
        # The leftover unit goes to the larger of the two tied remainders.
        assert vector.weights == (39321, 19661, 6553)
        assert sum(vector.weights) == CHAIN_WEIGHT_UNITS

    def test_a_zero_unit_target_never_collects_a_rounding_crumb(self) -> None:
        """A miner scored zero must be submitted at zero, not at one."""
        vector = to_u16_chain_vector(
            allocations(
                ("5Alice", 0, 1),
                ("5Bob", 1, 1),
                ("5Carol", 2, 1),
                ("5Dave", 3, 0),
                ("5Eve", 4, 0),
            )
        )
        assert vector.weights[3] == 0
        assert vector.weights[4] == 0
        assert sum(vector.weights) == CHAIN_WEIGHT_UNITS

    def test_ties_break_deterministically_on_hotkey(self) -> None:
        """The crumb must land on the same target on every validator."""
        first = to_u16_chain_vector(allocations(("5Zeta", 0, 1), ("5Alpha", 1, 1), ("5Mu", 2, 1)))
        second = to_u16_chain_vector(allocations(("5Zeta", 0, 1), ("5Alpha", 1, 1), ("5Mu", 2, 1)))
        assert first.weights == second.weights
        # 65535 = 3 * 21845 exactly, so nothing is left over here.
        assert first.weights == (21845, 21845, 21845)

    def test_the_leftover_crumb_goes_to_the_largest_remainder(self) -> None:
        vector = to_u16_chain_vector(allocations(("5Alice", 0, 1), ("5Bob", 1, 1)))
        # 65535 is odd: one side must take the extra unit, and it is decided by
        # hotkey order once the remainders tie.
        assert sorted(vector.weights) == [32767, 32768]
        assert vector.weights[0] == 32768

    def test_hotkeys_stay_aligned_with_uids(self) -> None:
        vector = to_u16_chain_vector(
            allocations(("5Alice", 7, 1), ("5Bob", 3, 2), ("5Carol", 91, 3))
        )
        assert vector.uids == (7, 3, 91)
        assert vector.hotkeys == ("5Alice", "5Bob", "5Carol")

    def test_empty_allocation_is_left_for_the_runtime_to_route_to_burn(self) -> None:
        with pytest.raises(WeightSubmissionError, match="burn hotkey"):
            to_u16_chain_vector([])

    def test_all_zero_allocation_raises(self) -> None:
        with pytest.raises(WeightSubmissionError, match="nothing to submit"):
            to_u16_chain_vector(allocations(("5Alice", 0, 0), ("5Bob", 1, 0)))

    def test_negative_units_raise(self) -> None:
        with pytest.raises(WeightSubmissionError, match="negative units"):
            to_u16_chain_vector(allocations(("5Alice", 0, 10), ("5Bob", 1, -1)))


class TestChainWeightVector:
    def test_ragged_columns_are_rejected(self) -> None:
        with pytest.raises(WeightSubmissionError, match="column lengths disagree"):
            ChainWeightVector(uids=(0, 1), weights=(1,), hotkeys=("5Alice", "5Bob"))

    def test_a_duplicated_uid_is_rejected(self) -> None:
        """A duplicate is a lost payment, not a rounding crumb.

        The chain stores one weight per UID, so a second entry overwrites the
        first: one of the two targets receives a weight nobody computed. The
        vector still sums to 65535, so no total looks wrong and nothing
        downstream can notice.
        """
        with pytest.raises(DuplicateWeightTargetError, match=r"uid\(s\) \[4\]"):
            ChainWeightVector(uids=(4, 4), weights=(30_000, 35_535), hotkeys=("5Alice", "5Bob"))

    def test_a_duplicated_hotkey_is_rejected(self) -> None:
        with pytest.raises(DuplicateWeightTargetError, match="hotkey"):
            ChainWeightVector(uids=(4, 9), weights=(30_000, 35_535), hotkeys=("5Alice", "5Alice"))

    def test_the_duplicate_error_carries_its_own_code(self) -> None:
        with pytest.raises(DuplicateWeightTargetError) as excinfo:
            ChainWeightVector(uids=(4, 4), weights=(1, 2), hotkeys=("5A", "5B"))
        assert excinfo.value.code == "chain.duplicate_weight_target"
