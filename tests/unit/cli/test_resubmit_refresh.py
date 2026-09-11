"""A mirror that fell behind catches up on its next re-post.

`cmd_resubmit` replayed whatever vector the last `validator run` wrote. If that
run fired before the provider published -- which is what a provider retrying a
failed cycle causes -- the mirror wrote nothing and spent the next day posting
the *previous* day's vector. That happened on netuid 108 on 2026-09-11 to 22.5%
of validating stake.

Re-reading the day already held would not help: a published record is immutable.
Only a newer day is worth looking for, and a mirror that is already current must
not spend a request looking.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair

from prometheon.chain.metagraph import MetagraphView
from prometheon.cli import validator as cli
from prometheon.dbclient.auth import sign_evaluation
from prometheon.dbclient.models import EvaluationSubmission, MinerResult, ModelStatus

pytestmark = pytest.mark.unit

DAY = dt.date(2026, 9, 10)
NETUID = 108
SNAP = "a" * 64
CORPUS = "b" * 64
BURN = "5Burn"


def keypair(seed: int) -> Keypair:
    return Keypair.create_from_seed(f"0x{seed:064x}")


def miner(hotkey: str, uid: int, weight: int) -> MinerResult:
    return MinerResult(
        hotkey=hotkey,
        uid=uid,
        weight=weight,
        dataset_submitted_count=0,
        dataset_valid_count=0,
        dataset_score_micro=0,
        model_status=ModelStatus.EVALUATED,
        model_items_scored=1,
        model_correct_count=1,
        raw_accuracy_bp=10_000,
        mean_total_tokens_milli=0,
        efficiency_penalty_bp=0,
        model_score_bp=0,
        model_rank=1,
    )


def record(provider: Keypair, *, weight: int, day: dt.date = DAY) -> EvaluationSubmission:
    return sign_evaluation(
        provider,
        EvaluationSubmission(
            date=day,
            netuid=NETUID,
            validator_hotkey=provider.ss58_address,
            scoring_version="prometheon-scoring/2.2",
            policy_version="2026-08-07",
            snapshot_content_hash=SNAP,
            corpus_content_hash=CORPUS,
            labelled_test_count=10,
            labelled_production_count=2,
            results=(miner("5Miner", 7, weight),),
            burned_weight=0,
            started_at=1,
            completed_at=2,
        ),
    )


def config(provider: str) -> SimpleNamespace:
    return SimpleNamespace(
        chain=SimpleNamespace(netuid=NETUID),
        db=SimpleNamespace(
            base_url="https://db.invalid", request_timeout_seconds=1.0, max_retries=1
        ),
        scoring=SimpleNamespace(score_provider=provider),
    )


def patch_chain(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    view = MetagraphView(
        netuid=NETUID,
        block=1,
        uids=(0,),
        hotkeys=(provider,),
        stake_rao=(1,),
        validator_permit=(True,),
    )
    monkeypatch.setattr(cli.chain, "sync_metagraph_view", lambda *a, **k: view)


def fake_db(
    monkeypatch: pytest.MonkeyPatch,
    published: dict[dt.date, object],
    *,
    raises: bool = False,
) -> list[dt.date]:
    """Stand in for the DB layer. Returns the list of days actually requested."""
    asked: list[dt.date] = []

    class FakeDb:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> FakeDb:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def get_evaluation(self, day: dt.date, _provider: str) -> object:
            asked.append(day)
            if raises:
                raise TimeoutError("db layer unreachable")
            if day not in published:
                raise LookupError(f"no record for {day}")
            return published[day]

        def get_snapshot(self, _day: dt.date) -> object:
            return SimpleNamespace(content_hash=SNAP)

    monkeypatch.setattr(cli, "DbClient", FakeDb)
    return asked


def call(cfg: SimpleNamespace, *, stored: dt.date, today: dt.date):
    return cli._refresh_mirrored(
        cfg,
        stored_day=stored,
        wallet=SimpleNamespace(hotkey=keypair(2)),
        subtensor=object(),
        burn_hotkey=BURN,
        today=today,
    )


def test_a_mirror_that_is_current_asks_for_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The common case must cost no request at all.

    A mirror holding yesterday's vector is up to date: nothing newer can exist,
    because a cycle scores the day before it runs. If this probed anyway it
    would add 48 pointless fetches a day per mirror, which is what made the
    first version of this not worth shipping.
    """
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    asked = fake_db(monkeypatch, {})
    assert call(config(provider.ss58_address), stored=DAY, today=DAY + dt.timedelta(days=1)) is None
    assert asked == []


def test_a_mirror_a_day_behind_catches_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """The uid 7 case: stuck on 09-09 because 09-10 was published late."""
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    behind = DAY - dt.timedelta(days=1)
    fake_db(monkeypatch, {DAY: record(provider, weight=4242, day=DAY)})
    found = call(config(provider.ss58_address), stored=behind, today=DAY + dt.timedelta(days=1))
    assert found is not None
    day, weights = found
    assert day == DAY
    assert weights == {"5Miner": 4242}


def test_the_newest_day_wins_and_older_ones_are_never_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    older, newer = DAY - dt.timedelta(days=1), DAY
    asked = fake_db(
        monkeypatch,
        {
            older: record(provider, weight=1, day=older),
            newer: record(provider, weight=2, day=newer),
        },
    )
    found = call(
        config(provider.ss58_address),
        stored=DAY - dt.timedelta(days=3),
        today=DAY + dt.timedelta(days=1),
    )
    assert found is not None and found[0] == newer
    assert asked == [newer], "an older day must not be fetched once a newer one verified"


def test_a_day_the_provider_has_not_published_yet_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every mirror sees this daily, between midnight and the provider's publish."""
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    fake_db(monkeypatch, {})
    assert (
        call(
            config(provider.ss58_address),
            stored=DAY - dt.timedelta(days=1),
            today=DAY + dt.timedelta(days=1),
        )
        is None
    )


def test_a_db_layer_that_is_down_falls_back_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The re-post must survive the fetch failing.

    Weights stop counting toward consensus once activity_cutoff passes, so a
    mirror that refuses to submit because it could not reach the DB layer costs
    its miners far more than re-posting a day-old vector does.
    """
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    fake_db(monkeypatch, {}, raises=True)
    assert (
        call(
            config(provider.ss58_address),
            stored=DAY - dt.timedelta(days=1),
            today=DAY + dt.timedelta(days=1),
        )
        is None
    )


def test_a_record_signed_by_the_wrong_hotkey_is_refused_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning score_provider means a substituted record is not submitted."""
    pinned, impostor = keypair(1), keypair(9)
    patch_chain(monkeypatch, pinned.ss58_address)
    fake_db(monkeypatch, {DAY: record(impostor, weight=999, day=DAY)})
    assert (
        call(
            config(pinned.ss58_address),
            stored=DAY - dt.timedelta(days=1),
            today=DAY + dt.timedelta(days=1),
        )
        is None
    )


def test_the_lookback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirror down for a month must not fetch a month of days every half hour."""
    provider = keypair(1)
    patch_chain(monkeypatch, provider.ss58_address)
    asked = fake_db(monkeypatch, {})
    call(
        config(provider.ss58_address),
        stored=DAY - dt.timedelta(days=400),
        today=DAY + dt.timedelta(days=1),
    )
    assert len(asked) == cli.MIRROR_LOOKBACK_DAYS
