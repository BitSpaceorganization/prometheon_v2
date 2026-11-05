"""Builders and constants for the hub tests: one day, one validator, two miners."""

from __future__ import annotations

import datetime as dt
from typing import Any

from bittensor_wallet import Keypair

from prometheon.scoring.dedupe import normalized_sha256

NETUID = 481
DAY = dt.date(2026, 11, 5)
#: 12:00 UTC on the day after: the epoch exists and the URLs are fresh.
NOW = int(dt.datetime(2026, 11, 6, 12, 0, tzinfo=dt.timezone.utc).timestamp())

VALIDATOR = Keypair.create_from_uri("//HubValidator")
MINER_A = Keypair.create_from_uri("//HubMinerA")
MINER_B = Keypair.create_from_uri("//HubMinerB")
COLD_A = Keypair.create_from_uri("//HubColdA")
COLD_B = Keypair.create_from_uri("//HubColdB")


class Clock:
    def __init__(self, now: int = NOW) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def miner_row(
    hotkey: str, uid: int, coldkey: str, verified: int = 14, members: int = 40
) -> dict[str, Any]:
    return {
        "hotkey": hotkey,
        "uid": uid,
        "coldkey": coldkey,
        "verified_active_members": verified,
        "members": members,
        "eligible": verified >= 10 and members >= 30,
    }


def feed_test_line(
    item_id: str,
    hotkey: str,
    body: str,
    *,
    claimed: bool = True,
    at: str = "2026-11-05T12:00:00.000Z",
    **overrides: Any,
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "id": item_id,
        "miner_hotkey": hotkey,
        "author": f"anon-{item_id}",
        "claimed_violating": claimed,
        "body": body,
        "normalized_sha256": normalized_sha256(body),
        "created_at": at,
        "verified": True,
    }
    line.update(overrides)
    return line


def production_line(
    item_id: str, body: str, *, at: str = "2026-11-05T08:00:00Z", **extra: Any
) -> dict[str, Any]:
    return {"id": item_id, "body": body, "created_at": at, **extra}


def standard_miners() -> list[dict[str, Any]]:
    return [
        miner_row(VALIDATOR.ss58_address, 0, VALIDATOR.ss58_address, verified=0, members=0),
        miner_row(MINER_A.ss58_address, 1, COLD_A.ss58_address),
        miner_row(MINER_B.ss58_address, 2, COLD_B.ss58_address),
    ]
