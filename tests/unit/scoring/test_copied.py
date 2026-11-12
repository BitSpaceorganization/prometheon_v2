"""A copied record is checked for provenance before it is ever submitted."""

from __future__ import annotations

import datetime as dt

import pytest
from bittensor_wallet import Keypair

from prometheon.chain.metagraph import MetagraphView
from prometheon.hub.auth import sign_evaluation
from prometheon.hub.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.scoring.copied import CopyRefusedError, verify_copied_record
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.unit

DAY = dt.date(2026, 8, 24)
NETUID = 108
HASH_A = "a" * 64
HASH_B = "b" * 64


BURN = "5Burn"


def keypair(seed: int) -> Keypair:
    return Keypair.create_from_seed(f"0x{seed:064x}")


def result(hotkey: str, uid: int, weight: int) -> MinerResult:
    return MinerResult(
        hotkey=hotkey,
        uid=uid,
        weight=weight,
        dataset_submitted_count=0,
        dataset_valid_count=0,
        dataset_violating_confirmed=0,
        dataset_compliant_confirmed=0,
        dataset_score_micro=0,
        model_status=ModelStatus.UNREACHABLE,
        model_items_scored=0,
        model_correct_count=0,
        raw_accuracy_bp=0,
        model_score_bp=0,
        model_rank=None,
    )


def record(owner: Keypair, **overrides: object) -> EvaluationSubmission:
    fields: dict[str, object] = {
        "date": DAY,
        "netuid": NETUID,
        "validator_hotkey": owner.ss58_address,
        "scoring_version": SCORING_VERSION,
        "policy_version": "2026-08-07",
        "epoch_content_hash": HASH_A,
        "corpus_content_hash": HASH_B,
        "labelled_test_count": 10,
        "labelled_production_count": 2,
        "results": (result("5Miner", 7, 1000),),
        "burned_weight": 500,
        "started_at": 1,
        "completed_at": 2,
    }
    fields.update(overrides)
    return sign_evaluation(owner, EvaluationSubmission(**fields))  # type: ignore[arg-type]


def metagraph(owner: str, *, permit: bool = True, registered: bool = True) -> MetagraphView:
    hotkeys = (owner,) if registered else ("5Somebody",)
    return MetagraphView(
        netuid=NETUID,
        block=1,
        uids=(0,),
        hotkeys=hotkeys,
        stake_rao=(1,),
        validator_permit=(permit,),
    )


def test_a_well_formed_record_yields_its_weights() -> None:
    owner = keypair(1)
    weights = verify_copied_record(
        record(owner),
        owner=owner.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(owner.ss58_address),
        burn_hotkey=BURN,
        epoch_content_hash=HASH_A,
    )
    # The burn is part of the vector. Asserting the miner rows alone is what
    # let a dropped burn ship: every provenance check passed, and the only
    # symptom was every miner silently inflated by the burn fraction.
    assert weights == {"5Miner": 1000, BURN: 500}


def test_a_record_from_another_validator_is_refused() -> None:
    """Copying copy_from and no other is what stops the hub choosing whom to trust."""
    other = keypair(2)
    with pytest.raises(CopyRefusedError, match="and no other hotkey"):
        verify_copied_record(
            record(other),
            owner=keypair(1).ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(other.ss58_address),
            burn_hotkey=BURN,
        )


def test_a_tampered_record_fails_its_signature() -> None:
    owner = keypair(1)
    signed = record(owner)
    forged = signed.model_copy(update={"results": (result("5Attacker", 9, 999_999),)})
    with pytest.raises(CopyRefusedError, match="signature does not verify"):
        verify_copied_record(
            forged,
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address),
            burn_hotkey=BURN,
        )


def test_the_wrong_day_or_netuid_is_refused() -> None:
    owner = keypair(1)
    with pytest.raises(CopyRefusedError, match="is for 2026-08-23"):
        verify_copied_record(
            record(owner, date=dt.date(2026, 8, 23)),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address),
            burn_hotkey=BURN,
        )
    with pytest.raises(CopyRefusedError, match="netuid=1"):
        verify_copied_record(
            record(owner, netuid=1),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address),
            burn_hotkey=BURN,
        )


def test_an_owner_validator_without_a_permit_is_refused() -> None:
    owner = keypair(1)
    with pytest.raises(CopyRefusedError, match="no validator permit"):
        verify_copied_record(
            record(owner),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address, permit=False),
            burn_hotkey=BURN,
        )
    with pytest.raises(CopyRefusedError, match="not registered"):
        verify_copied_record(
            record(owner),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address, registered=False),
            burn_hotkey=BURN,
        )


def test_a_record_naming_a_different_epoch_is_refused() -> None:
    """Catches being served a stale day, or a record for another day's content."""
    owner = keypair(1)
    with pytest.raises(CopyRefusedError, match="describe different days"):
        verify_copied_record(
            record(owner),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address),
            burn_hotkey=BURN,
            epoch_content_hash="c" * 64,
        )


def test_the_vector_sums_to_what_the_owner_validator_submitted() -> None:
    """The check that would have caught the dropped burn.

    Provenance can pass on a vector the owner never sent. Asserting the
    *total* is what ties the copied vector to the original: drop the burn and
    the sum falls short, and every miner is silently renormalised upward by the
    burn fraction -- 20% burn became a 1.25x on every miner in production, with
    vtrust at 0.8 as the only visible symptom.
    """
    owner = keypair(1)
    signed = record(owner)
    weights = verify_copied_record(
        signed,
        owner=owner.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(owner.ss58_address),
        burn_hotkey=BURN,
        epoch_content_hash=HASH_A,
    )
    expected = sum(r.weight for r in signed.results) + signed.burned_weight
    assert sum(weights.values()) == expected
    assert weights[BURN] == signed.burned_weight


def test_a_record_that_burned_nothing_carries_no_burn_row() -> None:
    """Zero burn is not a row. An explicit 0 would still be a weight the
    owner did not assign, and `resolve_allocations` would carry it."""
    owner = keypair(1)
    signed = record(owner, burned_weight=0)
    weights = verify_copied_record(
        signed,
        owner=owner.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(owner.ss58_address),
        burn_hotkey=BURN,
        epoch_content_hash=HASH_A,
    )
    assert BURN not in weights


def test_an_owner_validator_that_is_also_a_miner_has_its_burn_summed() -> None:
    """The owner hotkey can earn as a miner too. Its row carries its earnings and
    the record carries its burn, so the two add and neither is counted twice."""
    owner = keypair(1)
    signed = record(owner, results=(result(BURN, 7, 300),))
    weights = verify_copied_record(
        signed,
        owner=owner.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(owner.ss58_address),
        burn_hotkey=BURN,
        epoch_content_hash=HASH_A,
    )
    assert weights[BURN] == 300 + signed.burned_weight


def test_a_record_computed_under_other_rules_is_refused() -> None:
    """v2 never compared the scoring version, so a record under other rules was
    copied without anyone noticing (B26)."""
    owner = keypair(1)
    with pytest.raises(CopyRefusedError, match="computed under"):
        verify_copied_record(
            record(owner, scoring_version="prometheon-scoring/2.1"),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(owner.ss58_address),
            burn_hotkey=BURN,
            epoch_content_hash=HASH_A,
        )


def test_zero_rows_are_left_out_of_the_vector() -> None:
    """A owner does not send a hotkey it gave nothing to."""
    owner = keypair(1)
    signed = record(owner, results=(result("5Miner", 7, 1000), result("5Idle", 8, 0)))
    weights = verify_copied_record(
        signed,
        owner=owner.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(owner.ss58_address),
        burn_hotkey=BURN,
        epoch_content_hash=HASH_A,
    )
    assert "5Idle" not in weights


class TestThePermitIsReadByHotkeyNotByUid:
    """S10: the permit column is in the SDK's row order, which need not be uid order.

    v2 indexed it with the owner's uid. With rows ``(uid 1, uid 0)`` that
    reads the other neuron's permit: a permitted owner is refused, and an
    unpermitted one is accepted.
    """

    @staticmethod
    def _rows(owner: str, *, owner_permit: bool) -> MetagraphView:
        # Row 0 holds uid 1 (the owner); row 1 holds uid 0 (someone else),
        # whose permit is the opposite of the owner's.
        return MetagraphView(
            netuid=NETUID,
            block=1,
            uids=(1, 0),
            hotkeys=(owner, "5Somebody"),
            stake_rao=(1, 1),
            validator_permit=(owner_permit, not owner_permit),
        )

    def _verify(self, permit: bool) -> dict[str, int]:
        owner = keypair(1)
        return verify_copied_record(
            record(owner),
            owner=owner.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=self._rows(owner.ss58_address, owner_permit=permit),
            burn_hotkey=BURN,
            epoch_content_hash=HASH_A,
        )

    def test_a_permitted_owner_validator_is_accepted(self) -> None:
        assert self._verify(permit=True) == {"5Miner": 1000, BURN: 500}

    def test_an_unpermitted_owner_validator_is_refused(self) -> None:
        with pytest.raises(CopyRefusedError, match="no validator permit"):
            self._verify(permit=False)
