"""A day on the fake hub.

Everything goes through ``httpx.MockTransport`` into :class:`FakeHub`, so the
client is exercised against signed epochs, real gzip files and expiring URLs
rather than stubbed responses.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from bittensor_wallet import Keypair
from tests.unit.hub.factories import (
    DAY,
    MINER_A,
    MINER_B,
    NETUID,
    VALIDATOR,
    Clock,
    feed_test_line,
    production_line,
    standard_miners,
)

from prometheon.hub.auth import CallerRole
from prometheon.hub.client import HubClient
from prometheon.hub.fake import FakeHub


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def hub(clock: Clock) -> FakeHub:
    fake = FakeHub(netuid=NETUID, now=clock)
    fake.register(VALIDATOR.ss58_address, CallerRole.VALIDATOR)
    fake.register(MINER_A.ss58_address, CallerRole.MINER)
    fake.publish_day(
        DAY,
        miners=standard_miners(),
        test_lines=[
            feed_test_line("t1", MINER_A.ss58_address, "you are all idiots", claimed=True),
            feed_test_line("t2", MINER_B.ss58_address, "lovely weather today", claimed=False),
        ],
        production_lines=[production_line("p1", "an ordinary post")],
    )
    return fake


@pytest.fixture
def make_client(hub: FakeHub, clock: Clock) -> Callable[..., HubClient]:
    def build(
        *, keypair: Keypair | None = VALIDATOR, base_url: str = "https://hub.invalid", **kwargs: Any
    ) -> HubClient:
        options: dict[str, Any] = {
            "epoch_keys": hub.epoch_keys,
            "transport": hub.transport,
            "now": clock,
            "sleep": lambda _seconds: None,
        }
        options.update(kwargs)
        return HubClient(base_url=base_url, netuid=NETUID, keypair=keypair, **options)

    return build
