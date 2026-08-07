"""Shared fixtures for the DB-layer client tests.

Everything here is in-process: real keypairs, real signatures, a real fake
service, and an ``httpx.MockTransport``. No socket is opened and no file is
touched.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from bittensor_wallet import Keypair
from tests.unit.dbclient.factories import (
    BASE_URL,
    CYCLE_AT,
    DAY,
    NETUID,
    Clock,
    make_eligible,
    make_production_items,
    make_test_items,
)
from tests.unit.dbclient.factories import make_commitment as build_commitment

from prometheon.dbclient import CallerRole, DbClient, FakeDbLayer


@pytest.fixture
def validator_keypair() -> Keypair:
    return Keypair.create_from_uri("//Alice")


@pytest.fixture
def miner_keypair() -> Keypair:
    return Keypair.create_from_uri("//Bob")


@pytest.fixture
def stranger_keypair() -> Keypair:
    return Keypair.create_from_uri("//Charlie")


@pytest.fixture
def clock() -> Clock:
    return Clock(CYCLE_AT)


@pytest.fixture
def fake(clock: Clock, validator_keypair: Keypair, miner_keypair: Keypair) -> FakeDbLayer:
    layer = FakeDbLayer(netuid=NETUID, now=clock)
    layer.register(validator_keypair.ss58_address, CallerRole.VALIDATOR)
    layer.register(miner_keypair.ss58_address, CallerRole.MINER)
    return layer


@pytest.fixture
def seeded(fake: FakeDbLayer, miner_keypair: Keypair) -> FakeDbLayer:
    fake.seed_day(
        DAY,
        test_items=make_test_items(4, author=miner_keypair.ss58_address),
        production_items=make_production_items(7),
        eligible_miners=(make_eligible(miner_keypair.ss58_address),),
        model_commitments=(build_commitment(miner_keypair.ss58_address),),
    )
    return fake


@pytest.fixture
def client(fake: FakeDbLayer, validator_keypair: Keypair, clock: Clock) -> Iterator[DbClient]:
    with DbClient(
        base_url=BASE_URL,
        netuid=NETUID,
        keypair=validator_keypair,
        transport=fake.transport,
        now=clock,
        sleep=lambda _seconds: None,
    ) as db:
        yield db


@pytest.fixture
def miner_client(fake: FakeDbLayer, miner_keypair: Keypair, clock: Clock) -> Iterator[DbClient]:
    with DbClient(
        base_url=BASE_URL,
        netuid=NETUID,
        keypair=miner_keypair,
        transport=fake.transport,
        now=clock,
        sleep=lambda _seconds: None,
    ) as db:
        yield db
