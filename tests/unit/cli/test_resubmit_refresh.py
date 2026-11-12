"""A mirror that fell behind catches up on its next re-post.

`cmd_resubmit` replayed whatever vector the last `validator run` wrote. If that
run fired before the provider published -- which is what a provider retrying a
failed cycle causes -- the mirror wrote nothing and spent the next day posting
the *previous* day's vector. That happened on netuid 108 on 2026-09-11 to 22.5%
of validating stake.

Re-reading the day already held would not help: a published record is immutable.
Only a newer day is worth looking for, and a mirror that is already current must
not spend a request looking.

The hub here is the real client over the fake hub, so the records and epochs a
mirror reads are signed and verified exactly as they would be in production.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair

from prometheon.chain.metagraph import MetagraphView
from prometheon.cli import validator as cli
from prometheon.hub.auth import CallerRole, sign_evaluation
from prometheon.hub.client import HubClient
from prometheon.hub.epoch import epoch_content_hash
from prometheon.hub.fake import FakeHub
from prometheon.hub.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.unit

DAY = dt.date(2026, 11, 10)
NETUID = 481
NOW = int(dt.datetime(2026, 11, 11, 12, tzinfo=dt.timezone.utc).timestamp())
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
        dataset_violating_confirmed=0,
        dataset_compliant_confirmed=0,
        dataset_score_micro=0,
        model_status=ModelStatus.EVALUATED,
        model_items_scored=1,
        model_correct_count=1,
        raw_accuracy_bp=10_000,
        model_score_bp=0,
        model_rank=1,
    )


class World:
    """A fake hub with an epoch for every recent day, and a pinned provider."""

    def __init__(self, provider: Keypair) -> None:
        self.provider = provider
        self.hub = FakeHub(netuid=NETUID, now=lambda: NOW)
        self.hub.register(provider.ss58_address, CallerRole.VALIDATOR)
        self.epoch_hash: dict[dt.date, str] = {}
        for back in range(6):
            day = DAY - dt.timedelta(days=back)
            raw = self.hub.publish_day(
                day,
                miners=[
                    {
                        "hotkey": provider.ss58_address,
                        "uid": 0,
                        "coldkey": provider.ss58_address,
                        "verified_active_members": 0,
                        "members": 0,
                        "eligible": False,
                    }
                ],
            )
            self.epoch_hash[day] = epoch_content_hash(raw)

    def publish(self, day: dt.date, *, weight: int, signer: Keypair | None = None) -> None:
        record = sign_evaluation(
            signer or self.provider,
            EvaluationSubmission(
                date=day,
                netuid=NETUID,
                validator_hotkey=(signer or self.provider).ss58_address,
                scoring_version=SCORING_VERSION,
                policy_version="2026-08-07",
                epoch_content_hash=self.epoch_hash[day],
                corpus_content_hash="b" * 64,
                labelled_test_count=10,
                labelled_production_count=2,
                results=(miner("5Miner", 7, weight),),
                burned_weight=0,
                started_at=1,
                completed_at=2,
            ),
        )
        # Stored under the pinned provider whoever signed it, as a hub serving a
        # substituted record would.
        self.hub.results[(day, self.provider.ss58_address)] = record.model_dump_json().encode()

    def results_asked(self) -> list[str]:
        return [target for method, target in self.hub.request_log if "/results/" in target]


def install(monkeypatch: pytest.MonkeyPatch, world: World, *, down: bool = False) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if down:
            raise httpx.ConnectError("hub unreachable", request=request)
        return world.hub.handler(request)

    def client(config: Any, *, keypair: Any = None) -> HubClient:
        return HubClient(
            base_url="https://hub.invalid",
            netuid=NETUID,
            epoch_keys=world.hub.epoch_keys,
            keypair=keypair,
            transport=httpx.MockTransport(handler),
            max_retries=0,
            sleep=lambda _seconds: None,
        )

    view = MetagraphView(
        netuid=NETUID,
        block=1,
        uids=(0,),
        hotkeys=(world.provider.ss58_address,),
        stake_rao=(1,),
        validator_permit=(True,),
    )
    monkeypatch.setattr(cli, "_hub_client", client)
    monkeypatch.setattr(cli.chain, "sync_metagraph_view", lambda *a, **k: view)


def call(world: World, *, stored: dt.date, today: dt.date) -> Any:
    config = SimpleNamespace(
        chain=SimpleNamespace(netuid=NETUID),
        scoring=SimpleNamespace(score_provider=world.provider.ss58_address),
    )
    return cli._refresh_mirrored(
        config,  # type: ignore[arg-type]
        stored_day=stored,
        wallet=SimpleNamespace(hotkey=keypair(2)),
        subtensor=object(),
        burn_hotkey=BURN,
        today=today,
    )


def test_a_mirror_that_is_current_asks_for_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirror holding yesterday's vector is up to date: nothing newer exists."""
    world = World(keypair(1))
    install(monkeypatch, world)
    assert call(world, stored=DAY, today=DAY + dt.timedelta(days=1)) is None
    assert world.hub.request_log == []


def test_a_mirror_a_day_behind_catches_up(monkeypatch: pytest.MonkeyPatch) -> None:
    world = World(keypair(1))
    install(monkeypatch, world)
    world.publish(DAY, weight=4242)
    found = call(world, stored=DAY - dt.timedelta(days=1), today=DAY + dt.timedelta(days=1))
    assert found == (DAY, {"5Miner": 4242})


def test_the_newest_day_wins_and_older_ones_are_never_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(keypair(1))
    install(monkeypatch, world)
    older = DAY - dt.timedelta(days=1)
    world.publish(older, weight=1)
    world.publish(DAY, weight=2)
    found = call(world, stored=DAY - dt.timedelta(days=3), today=DAY + dt.timedelta(days=1))
    assert found is not None and found[0] == DAY
    assert len(world.results_asked()) == 1
    assert DAY.isoformat() in world.results_asked()[0]


def test_a_day_the_provider_has_not_published_yet_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(keypair(1))
    install(monkeypatch, world)
    assert call(world, stored=DAY - dt.timedelta(days=1), today=DAY + dt.timedelta(days=1)) is None


def test_a_hub_that_is_down_falls_back_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights stop counting once activity_cutoff passes, so a mirror that refuses to
    re-post because the hub is unreachable costs its miners more than a day-old vector."""
    world = World(keypair(1))
    install(monkeypatch, world, down=True)
    assert call(world, stored=DAY - dt.timedelta(days=1), today=DAY + dt.timedelta(days=1)) is None


def test_a_record_signed_by_the_wrong_hotkey_is_refused_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = World(keypair(1))
    install(monkeypatch, world)
    world.publish(DAY, weight=999, signer=keypair(9))
    assert call(world, stored=DAY - dt.timedelta(days=1), today=DAY + dt.timedelta(days=1)) is None


def test_the_lookback_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mirror down for a month must not fetch a month of days every half hour."""
    world = World(keypair(1))
    install(monkeypatch, world)
    call(world, stored=DAY - dt.timedelta(days=400), today=DAY + dt.timedelta(days=1))
    assert len(world.results_asked()) == cli.MIRROR_LOOKBACK_DAYS
