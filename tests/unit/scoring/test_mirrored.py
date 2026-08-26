"""A mirrored record is checked for provenance before it is ever submitted."""

from __future__ import annotations

import datetime as dt

import pytest
from bittensor_wallet import Keypair

from prometheon.chain.metagraph import MetagraphView
from prometheon.dbclient.auth import sign_evaluation
from prometheon.dbclient.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.scoring.mirrored import MirroredScoreError, verify_mirrored

pytestmark = pytest.mark.unit

DAY = dt.date(2026, 8, 24)
NETUID = 108
HASH_A = "a" * 64
HASH_B = "b" * 64


def keypair(seed: int) -> Keypair:
    return Keypair.create_from_seed(f"0x{seed:064x}")


def result(hotkey: str, uid: int, weight: int) -> MinerResult:
    return MinerResult(
        hotkey=hotkey,
        uid=uid,
        weight=weight,
        dataset_submitted_count=0,
        dataset_valid_count=0,
        dataset_score_micro=0,
        model_status=ModelStatus.UNREACHABLE,
        model_items_scored=0,
        model_correct_count=0,
        raw_accuracy_bp=0,
        mean_total_tokens_milli=0,
        efficiency_penalty_bp=0,
        model_score_bp=0,
        model_rank=None,
    )


def record(provider: Keypair, **overrides: object) -> EvaluationSubmission:
    fields: dict[str, object] = {
        "date": DAY,
        "netuid": NETUID,
        "validator_hotkey": provider.ss58_address,
        "scoring_version": "prometheon-scoring/2.1",
        "policy_version": "2026-08-07",
        "snapshot_content_hash": HASH_A,
        "corpus_content_hash": HASH_B,
        "labelled_test_count": 10,
        "labelled_production_count": 2,
        "results": (result("5Miner", 7, 1000),),
        "burned_weight": 500,
        "started_at": 1,
        "completed_at": 2,
    }
    fields.update(overrides)
    return sign_evaluation(provider, EvaluationSubmission(**fields))  # type: ignore[arg-type]


def metagraph(provider: str, *, permit: bool = True, registered: bool = True) -> MetagraphView:
    hotkeys = (provider,) if registered else ("5Somebody",)
    return MetagraphView(
        netuid=NETUID,
        block=1,
        uids=(0,),
        hotkeys=hotkeys,
        stake_rao=(1,),
        validator_permit=(permit,),
    )


def test_a_well_formed_record_yields_its_weights() -> None:
    provider = keypair(1)
    weights = verify_mirrored(
        record(provider),
        provider=provider.ss58_address,
        day=DAY,
        netuid=NETUID,
        metagraph=metagraph(provider.ss58_address),
        snapshot_content_hash=HASH_A,
    )
    assert weights == {"5Miner": 1000}


def test_a_record_from_another_validator_is_refused() -> None:
    """Pinning the provider is what stops the DB layer choosing who to trust."""
    other = keypair(2)
    with pytest.raises(MirroredScoreError, match="mirrors"):
        verify_mirrored(
            record(other),
            provider=keypair(1).ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(other.ss58_address),
        )


def test_a_tampered_record_fails_its_signature() -> None:
    provider = keypair(1)
    signed = record(provider)
    forged = signed.model_copy(update={"results": (result("5Attacker", 9, 999_999),)})
    with pytest.raises(MirroredScoreError, match="signature does not verify"):
        verify_mirrored(
            forged,
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address),
        )


def test_the_wrong_day_or_netuid_is_refused() -> None:
    provider = keypair(1)
    with pytest.raises(MirroredScoreError, match="is for 2026-08-23"):
        verify_mirrored(
            record(provider, date=dt.date(2026, 8, 23)),
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address),
        )
    with pytest.raises(MirroredScoreError, match="netuid=1"):
        verify_mirrored(
            record(provider, netuid=1),
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address),
        )


def test_a_provider_without_a_permit_is_refused() -> None:
    provider = keypair(1)
    with pytest.raises(MirroredScoreError, match="no validator permit"):
        verify_mirrored(
            record(provider),
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address, permit=False),
        )
    with pytest.raises(MirroredScoreError, match="not registered"):
        verify_mirrored(
            record(provider),
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address, registered=False),
        )


def test_a_record_describing_a_different_snapshot_is_refused() -> None:
    """Catches being served a stale day, or a different corpus for the same day."""
    provider = keypair(1)
    with pytest.raises(MirroredScoreError, match="different corpora"):
        verify_mirrored(
            record(provider),
            provider=provider.ss58_address,
            day=DAY,
            netuid=NETUID,
            metagraph=metagraph(provider.ss58_address),
            snapshot_content_hash="c" * 64,
        )
