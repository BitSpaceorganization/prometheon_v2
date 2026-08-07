"""Constants and builders shared by the DB-layer tests.

Separate from ``conftest.py`` so test modules can import them by name without
pytest's conftest loading giving the module two identities.
"""

from __future__ import annotations

import datetime as dt
import itertools

from prometheon.dbclient import (
    EligibleMiner,
    ModelCommitment,
    ProductionContentItem,
    TestContentItem,
)

NETUID = 42

#: The day being scored, and the instant the 04:00 UTC cycle runs on the day
#: after it.
DAY = dt.date(2026, 8, 5)
CYCLE_AT = int(dt.datetime(2026, 8, 6, 4, 0, tzinfo=dt.timezone.utc).timestamp())

BASE_URL = "https://db.invalid"


class Clock:
    """A clock a test moves by hand."""

    def __init__(self, now: int) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += seconds


class RecordingSleep:
    """Stands in for ``time.sleep`` and remembers what it was asked to wait."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


_SEQUENCE = itertools.count(1)


def make_test_items(count: int, *, author: str, prefix: str = "t") -> tuple[TestContentItem, ...]:
    return tuple(
        TestContentItem(
            id=f"{prefix}{index}",
            content=f"test content {index}",
            author_hotkey=author,
            submitted_at=CYCLE_AT - 3600 * index,
        )
        for index in range(1, count + 1)
    )


def make_production_items(count: int, *, prefix: str = "p") -> tuple[ProductionContentItem, ...]:
    return tuple(
        ProductionContentItem(
            id=f"{prefix}{index}",
            content=f"production content {index}",
            observed_at=CYCLE_AT - 60 * index,
        )
        for index in range(1, count + 1)
    )


def make_commitment(hotkey: str, *, revision: str | None = None) -> ModelCommitment:
    serial = next(_SEQUENCE)
    return ModelCommitment(
        hotkey=hotkey,
        hf_repo=f"miner/model-{serial}",
        revision_sha=revision or f"{serial:040x}",
        chute_id=f"chute-{serial}",
        committed_at=CYCLE_AT - 86_400,
        block=1_000 + serial,
    )


def make_eligible(hotkey: str, *, qualified_user_count: int = 9) -> EligibleMiner:
    return EligibleMiner(hotkey=hotkey, qualified_user_count=qualified_user_count)
