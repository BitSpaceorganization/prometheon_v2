"""Behaviour of the metagraph snapshot and hotkey resolution."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from prometheon.chain.metagraph import MetagraphView
from prometheon.errors import SubtensorError

pytestmark = pytest.mark.unit

HOTKEYS = ("5Alice", "5Bob", "5Carol")


def view(**overrides: Any) -> MetagraphView:
    fields: dict[str, Any] = {
        "netuid": 42,
        "block": 1000,
        "uids": (0, 1, 2),
        "hotkeys": HOTKEYS,
        "stake_rao": (0, 5_000_000_000, 1),
        "validator_permit": (False, True, False),
    }
    fields.update(overrides)
    return MetagraphView(**fields)


class TestResolution:
    def test_uid_for_a_registered_hotkey(self) -> None:
        assert view().uid_for("5Bob") == 1

    def test_uid_for_an_unregistered_hotkey_is_none(self) -> None:
        assert view().uid_for("5Mallory") is None

    def test_require_uid_raises_rather_than_returning_none(self) -> None:
        with pytest.raises(SubtensorError, match="not registered on netuid=42"):
            view().require_uid("5Mallory")

    def test_resolution_survives_non_contiguous_uids(self) -> None:
        """UIDs are recycled, so a live subnet's table is not a range."""
        snapshot = view(uids=(7, 3, 91))
        assert snapshot.uid_for("5Alice") == 7
        assert snapshot.uid_for("5Carol") == 91
        assert snapshot.hotkey_for(3) == "5Bob"
        assert snapshot.hotkey_for(0) is None

    def test_columns_stay_aligned_to_the_hotkey_not_the_uid(self) -> None:
        snapshot = view(uids=(7, 3, 91))
        assert snapshot.stake_rao_for("5Bob") == 5_000_000_000
        assert snapshot.has_validator_permit("5Bob") is True
        assert snapshot.has_validator_permit("5Carol") is False

    def test_unknown_hotkey_reads_as_absent_not_as_zero_stake(self) -> None:
        assert view().stake_rao_for("5Mallory") is None
        assert view().has_validator_permit("5Mallory") is False

    def test_validator_uids(self) -> None:
        assert view(uids=(7, 3, 91)).validator_uids() == (3,)

    def test_partition_preserves_input_order(self) -> None:
        registered, unregistered = view().partition_by_registration(
            ["5Carol", "5Mallory", "5Alice", "5Eve"]
        )
        assert registered == ["5Carol", "5Alice"]
        assert unregistered == ["5Mallory", "5Eve"]

    def test_length_is_the_neuron_count(self) -> None:
        assert len(view()) == 3
        assert len(MetagraphView(netuid=1)) == 0

    def test_the_view_is_frozen(self) -> None:
        snapshot = view()
        with pytest.raises(ValidationError):
            snapshot.block = 2000  # type: ignore[misc]


class TestConsistency:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("uids", (0, 1)),
            ("stake_rao", (0, 1, 2, 3)),
            ("validator_permit", (True,)),
        ],
    )
    def test_a_ragged_column_is_rejected(self, field: str, value: tuple[Any, ...]) -> None:
        with pytest.raises(ValidationError, match="entries"):
            view(**{field: value})

    def test_duplicate_uids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="uids are not unique"):
            view(uids=(0, 1, 1))

    def test_duplicate_hotkeys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="hotkeys are not unique"):
            view(hotkeys=("5Alice", "5Bob", "5Bob"))

    def test_negative_stake_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="negative stake"):
            view(stake_rao=(0, -1, 1))


class FakeMetagraph:
    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class FakeBalance:
    def __init__(self, rao: int) -> None:
        self.rao = rao


class TestFromSdkMetagraph:
    def test_reads_the_common_shape(self) -> None:
        raw = FakeMetagraph(
            hotkeys=list(HOTKEYS),
            uids=[0, 1, 2],
            total_stake=[0.0, 5.0, 0.000000001],
            validator_permit=[False, True, False],
            block=99,
        )
        snapshot = MetagraphView.from_sdk_metagraph(raw, netuid=42)
        assert snapshot.block == 99
        assert snapshot.hotkeys == HOTKEYS
        assert snapshot.stake_rao == (0, 5_000_000_000, 1)

    def test_tao_floats_become_exact_rao_integers(self) -> None:
        """No float may survive into anything that later gets canonicalised."""
        raw = FakeMetagraph(hotkeys=["5Alice"], total_stake=[1.1])
        snapshot = MetagraphView.from_sdk_metagraph(raw, netuid=1)
        assert snapshot.stake_rao == (1_100_000_000,)
        assert isinstance(snapshot.stake_rao[0], int)

    def test_balance_objects_are_read_without_going_through_a_float(self) -> None:
        raw = FakeMetagraph(hotkeys=["5Alice"], stake=[FakeBalance(123456789)])
        assert MetagraphView.from_sdk_metagraph(raw, netuid=1).stake_rao == (123456789,)

    def test_the_legacy_stake_attribute_is_accepted(self) -> None:
        raw = FakeMetagraph(hotkeys=["5Alice", "5Bob"], S=[1.0, 2.0])
        assert MetagraphView.from_sdk_metagraph(raw, netuid=1).stake_rao == (
            1_000_000_000,
            2_000_000_000,
        )

    def test_missing_uids_fall_back_to_position(self) -> None:
        raw = FakeMetagraph(hotkeys=list(HOTKEYS))
        assert MetagraphView.from_sdk_metagraph(raw, netuid=1).uids == (0, 1, 2)

    def test_missing_stake_and_permits_do_not_fail_a_cycle(self) -> None:
        snapshot = MetagraphView.from_sdk_metagraph(FakeMetagraph(hotkeys=list(HOTKEYS)), netuid=1)
        assert snapshot.stake_rao == (0, 0, 0)
        assert snapshot.validator_permit == (False, False, False)

    @pytest.mark.parametrize(
        "raw",
        [
            FakeMetagraph(hotkeys=["5Alice", "5Bob"], uids=[0]),
            FakeMetagraph(hotkeys=["5Alice", "5Bob"], total_stake=[1.0]),
            FakeMetagraph(hotkeys=["5Alice", "5Bob"], validator_permit=[True]),
        ],
    )
    def test_a_short_column_is_never_papered_over(self, raw: FakeMetagraph) -> None:
        """Recovering by position would silently mis-attribute every later row."""
        with pytest.raises(SubtensorError, match="unexpected shape"):
            MetagraphView.from_sdk_metagraph(raw, netuid=42)

    def test_a_duplicate_hotkey_from_the_sdk_surfaces_as_a_chain_error(self) -> None:
        raw = FakeMetagraph(hotkeys=["5Alice", "5Alice"])
        with pytest.raises(SubtensorError, match="inconsistent"):
            MetagraphView.from_sdk_metagraph(raw, netuid=42)

    def test_an_empty_metagraph_is_a_valid_snapshot(self) -> None:
        snapshot = MetagraphView.from_sdk_metagraph(FakeMetagraph(hotkeys=[]), netuid=42)
        assert len(snapshot) == 0
        assert snapshot.uid_for("5Alice") is None
