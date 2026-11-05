"""The signed epoch: nothing in it is used before its signature verifies."""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError
from tests.unit.hub.factories import (
    COLD_A,
    DAY,
    MINER_A,
    MINER_B,
    NETUID,
    miner_row,
    standard_miners,
)

from prometheon.errors import EpochVerificationError
from prometheon.hub.epoch import (
    check_epoch_is_for,
    eligible_miners,
    epoch_content_hash,
    epoch_signing_bytes,
    is_eligible,
    paid_models,
    verify_epoch,
)
from prometheon.hub.fake import FakeHub
from prometheon.hub.models import EpochMiner
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.unit


def _model(hotkey: str, payment: str = "paid", fingerprint: str = "f" * 64) -> dict[str, Any]:
    return {
        "hotkey": hotkey,
        "repo": "org/guard",
        "revision": "a" * 40,
        "commit_block": 10,
        "fingerprint": fingerprint,
        "payment": payment,
    }


@pytest.fixture
def signed(hub: FakeHub) -> dict[str, Any]:
    return hub.publish_day(
        DAY,
        miners=standard_miners(),
        models=[_model(MINER_A.ss58_address), _model(MINER_B.ss58_address, "unpaid", "e" * 64)],
    )


class TestTheSignature:
    def test_a_signed_epoch_verifies_and_parses(self, hub: FakeHub, signed: dict[str, Any]) -> None:
        epoch = verify_epoch(signed, keys=hub.epoch_keys)
        assert epoch.day == DAY
        assert epoch.netuid == NETUID
        assert [model.hotkey for model in paid_models(epoch)] == [MINER_A.ss58_address]

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (("netuid",), 482),
            (("miners", 1, "verified_active_members"), 9999),
            (("models", 1, "payment"), "paid"),
            (("files", "test", "count"), 7),
        ],
    )
    def test_any_change_to_a_signed_field_fails(
        self, hub: FakeHub, signed: dict[str, Any], path: tuple[Any, ...], value: Any
    ) -> None:
        tampered = copy.deepcopy(signed)
        target: Any = tampered
        for step in path[:-1]:
            target = target[step]
        assert target[path[-1]] != value
        target[path[-1]] = value
        with pytest.raises(EpochVerificationError, match="does not verify"):
            verify_epoch(tampered, keys=hub.epoch_keys)

    def test_the_key_id_is_inside_the_signed_bytes(
        self, hub: FakeHub, signed: dict[str, Any]
    ) -> None:
        """Relabelling an epoch to another pinned key must not verify."""
        keys = {**hub.epoch_keys, "hub-other": hub.public_key}
        with pytest.raises(EpochVerificationError, match="does not verify"):
            verify_epoch({**signed, "key_id": "hub-other"}, keys=keys)

    def test_a_field_this_client_does_not_know_is_still_signed(self, hub: FakeHub) -> None:
        """Unknown fields are ignored when parsing, but never when verifying."""
        raw = hub.publish_day(DAY, miners=standard_miners(), extra={"built_at": 1})
        verify_epoch(raw, keys=hub.epoch_keys)
        tampered = {**raw, "built_at": 2}
        with pytest.raises(EpochVerificationError, match="does not verify"):
            verify_epoch(tampered, keys=hub.epoch_keys)

    def test_a_key_id_that_is_not_pinned_is_refused(
        self, hub: FakeHub, signed: dict[str, Any]
    ) -> None:
        with pytest.raises(EpochVerificationError, match="does not pin"):
            verify_epoch(signed, keys={"other-key": hub.public_key})

    def test_no_pinned_key_at_all_is_refused_with_the_fix(self, signed: dict[str, Any]) -> None:
        with pytest.raises(EpochVerificationError, match=r"\[hub.epoch_keys\]"):
            verify_epoch(signed, keys={})

    def test_a_different_key_under_the_same_id_is_refused(
        self, hub: FakeHub, signed: dict[str, Any]
    ) -> None:
        impostor = FakeHub(netuid=NETUID, signing_seed="11" * 32)
        with pytest.raises(EpochVerificationError, match="does not verify"):
            verify_epoch(signed, keys={hub.key_id: impostor.public_key})

    def test_the_0x_prefix_on_the_signature_is_optional(
        self, hub: FakeHub, signed: dict[str, Any]
    ) -> None:
        bare = {**signed, "signature": signed["signature"][2:]}
        verify_epoch(bare, keys=hub.epoch_keys)

    def test_a_malformed_signature_is_refused(self, hub: FakeHub, signed: dict[str, Any]) -> None:
        with pytest.raises(EpochVerificationError):
            verify_epoch({**signed, "signature": "0xnothex"}, keys=hub.epoch_keys)

    def test_a_float_anywhere_is_refused_rather_than_canonicalised(self, hub: FakeHub) -> None:
        raw = hub.publish_day(DAY, miners=standard_miners())
        with pytest.raises(EpochVerificationError, match="canonicalised"):
            verify_epoch({**raw, "frozen_block": 1.0}, keys=hub.epoch_keys)

    def test_something_that_is_not_an_object_is_refused(self, hub: FakeHub) -> None:
        with pytest.raises(EpochVerificationError, match="not a JSON object"):
            verify_epoch(["epoch"], keys=hub.epoch_keys)

    def test_the_content_hash_is_the_hash_of_the_signed_bytes(self, signed: dict[str, Any]) -> None:
        import hashlib

        assert epoch_content_hash(signed) == hashlib.sha256(epoch_signing_bytes(signed)).hexdigest()
        assert b'"signature"' not in epoch_signing_bytes(signed)
        assert b'"key_id"' in epoch_signing_bytes(signed)


class TestTheSchema:
    def test_another_schema_is_refused_after_the_signature_verifies(self, hub: FakeHub) -> None:
        raw = hub.publish_day(DAY, miners=standard_miners(), extra={"schema": "prometheon-epoch/2"})
        with pytest.raises(EpochVerificationError, match="prometheon-epoch/1"):
            verify_epoch(raw, keys=hub.epoch_keys)

    def test_a_hotkey_listed_twice_is_refused(self, hub: FakeHub) -> None:
        miners = [*standard_miners(), miner_row(MINER_A.ss58_address, 9, COLD_A.ss58_address)]
        raw = hub.publish_day(DAY, miners=miners)
        with pytest.raises(EpochVerificationError, match="appears twice"):
            verify_epoch(raw, keys=hub.epoch_keys)

    def test_two_models_for_one_hotkey_are_refused(self, hub: FakeHub) -> None:
        raw = hub.publish_day(
            DAY,
            miners=standard_miners(),
            models=[_model(MINER_A.ss58_address), _model(MINER_A.ss58_address, "unpaid")],
        )
        with pytest.raises(EpochVerificationError, match="two frozen models"):
            verify_epoch(raw, keys=hub.epoch_keys)

    def test_a_boolean_must_be_a_boolean(self, hub: FakeHub) -> None:
        miners = standard_miners()
        miners[1]["eligible"] = 1
        raw = hub.publish_day(DAY, miners=miners)
        with pytest.raises(EpochVerificationError, match="not one this client accepts"):
            verify_epoch(raw, keys=hub.epoch_keys)

    def test_a_miner_without_a_member_count_is_refused(self, hub: FakeHub) -> None:
        """Read as zero, a missing count would quietly burn both pools for the day."""
        miners = standard_miners()
        del miners[1]["members"]
        raw = hub.publish_day(DAY, miners=miners)
        with pytest.raises(EpochVerificationError, match="members"):
            verify_epoch(raw, keys=hub.epoch_keys)

    def test_more_verified_active_members_than_members_is_refused(self, hub: FakeHub) -> None:
        miners = standard_miners()
        miners[1]["members"] = 13
        raw = hub.publish_day(DAY, miners=miners)
        with pytest.raises(EpochVerificationError, match="14 verified active members but only 13"):
            verify_epoch(raw, keys=hub.epoch_keys)


class TestWhichDay:
    def test_the_right_epoch_passes(self, hub: FakeHub, signed: dict[str, Any]) -> None:
        epoch = verify_epoch(signed, keys=hub.epoch_keys)
        check_epoch_is_for(epoch, day=DAY, netuid=NETUID, scoring_version=SCORING_VERSION)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("day", DAY.replace(day=4), "asked for the epoch"),
            ("netuid", NETUID + 1, "netuid"),
            ("scoring_version", "prometheon-scoring/2.1", "frozen under"),
        ],
    )
    def test_another_day_subnet_or_rule_set_is_refused(
        self, hub: FakeHub, signed: dict[str, Any], field: str, value: Any, message: str
    ) -> None:
        epoch = verify_epoch(signed, keys=hub.epoch_keys)
        expected = {"day": DAY, "netuid": NETUID, "scoring_version": SCORING_VERSION}
        expected[field] = value
        with pytest.raises(EpochVerificationError, match=message):
            check_epoch_is_for(epoch, **expected)


class TestEligibility:
    """At least 10 verified active members AND at least 30 members in total."""

    FLOORS: ClassVar[dict[str, int]] = {"min_verified_active_members": 10, "min_members": 30}

    def _miner(self, verified: int, members: int, flag: bool | None = None) -> EpochMiner:
        row = miner_row(MINER_A.ss58_address, 1, COLD_A.ss58_address, verified, members)
        row["eligible"] = flag
        return EpochMiner.model_validate(row)

    @pytest.mark.parametrize(
        ("verified", "members", "expected"),
        [
            (10, 30, True),
            (9, 30, False),
            (10, 29, False),
            (14, 120, True),
            (29, 29, False),
            (0, 0, False),
        ],
    )
    def test_both_counts_must_reach_their_floor(
        self, verified: int, members: int, expected: bool
    ) -> None:
        assert is_eligible(self._miner(verified, members), **self.FLOORS) is expected

    @pytest.mark.parametrize("flag", [True, False, None])
    def test_the_hub_flag_decides_nothing(self, flag: bool | None) -> None:
        assert is_eligible(self._miner(14, 30, flag), **self.FLOORS)
        assert not is_eligible(self._miner(14, 29, flag), **self.FLOORS)

    def test_the_floors_are_the_validators_config(self) -> None:
        miner = self._miner(14, 40)
        assert not is_eligible(miner, min_verified_active_members=15, min_members=30)
        assert not is_eligible(miner, min_verified_active_members=10, min_members=41)
        assert is_eligible(miner, min_verified_active_members=14, min_members=40)

    def test_more_verified_active_members_than_members_cannot_be_built(self) -> None:
        with pytest.raises(ValidationError, match="only 9 members"):
            self._miner(10, 9)

    def test_eligible_miners_keeps_the_epoch_order(
        self, hub: FakeHub, signed: dict[str, Any]
    ) -> None:
        epoch = verify_epoch(signed, keys=hub.epoch_keys)
        assert [miner.hotkey for miner in eligible_miners(epoch, **self.FLOORS)] == [
            MINER_A.ss58_address,
            MINER_B.ss58_address,
        ]
