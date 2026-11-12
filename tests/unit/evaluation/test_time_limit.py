"""B28: one fixed time limit per model, and running out of it is not a score.

v3.0 counted every item left at the deadline as wrong, so the same model scored
lower on a slower validator, which then disagreed with the rest for a reason
that had nothing to do with the model. Now a model that does not finish within
``[evaluation] model_timeout_seconds`` is marked timed out, gets no score, and
its unfinished items count neither way.

The clock is injected: each judged batch advances it by a fixed step.
"""

from __future__ import annotations

from typing import Any

import pytest

from prometheon.evaluation.corpus import ItemSource, LabelledItem
from prometheon.evaluation.engine import ModerationResult, Verdict
from prometheon.evaluation.runner import MinerTarget, evaluate_miner

pytestmark = pytest.mark.unit

TARGET = MinerTarget(hotkey="5Miner", hf_repo="org/guard", hf_revision="a" * 40)

#: Ten items, half violating. With a batch size of 2 that is five batches.
ITEMS = [
    LabelledItem(
        item_id=f"p{index}", content="x", violating=index % 2 == 0, source=ItemSource.PRODUCTION
    )
    for index in range(10)
]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Engine:
    """Answers every item correctly; loading and each batch take set times."""

    def __init__(
        self,
        clock: Clock,
        *,
        load_seconds: float = 0.0,
        batch_seconds: float = 10.0,
        fail_batches: frozenset[int] = frozenset(),
    ) -> None:
        self.clock = clock
        self.load_seconds = load_seconds
        self.batch_seconds = batch_seconds
        self.fail_batches = fail_batches
        self.batches_asked = 0
        self.unloaded = False

    def load(self) -> None:
        self.clock.now += self.load_seconds

    def unload(self) -> None:
        self.unloaded = True

    def moderate(self, policy: str, items: Any) -> ModerationResult:
        index = self.batches_asked
        self.batches_asked += 1
        self.clock.now += self.batch_seconds
        if index in self.fail_batches:
            raise RuntimeError("CUDA out of memory")
        truth = {item.item_id: item.violating for item in ITEMS}
        return ModerationResult(
            verdicts=tuple(
                Verdict(item_id=item_id, violates=truth[item_id]) for item_id, _ in items
            )
        )


def _evaluate(engine: Engine, clock: Clock, deadline: float | None) -> Any:
    return evaluate_miner(
        engine=engine,  # type: ignore[arg-type]
        target=TARGET,
        items=ITEMS,
        policy="p",
        policy_version="v",
        batch_size=2,
        deadline_seconds=deadline,
        clock=clock,
    )


def test_a_model_that_finishes_in_time_is_scored_on_everything() -> None:
    clock = Clock()
    result = _evaluate(Engine(clock), clock, deadline=60)

    assert result.timed_out is False
    assert (result.correct, result.total) == (10, 10)
    assert result.unfinished_items == 0


def test_a_model_that_runs_out_of_time_stops_and_counts_nothing_unfinished() -> None:
    """Five batches of 10s against a 25s limit: after the third the clock reads
    30s, over the limit, so it stops with 6 items judged and 4 never reached."""
    clock = Clock()
    engine = Engine(clock)
    result = _evaluate(engine, clock, deadline=25)

    assert result.timed_out is True
    assert engine.batches_asked == 3
    assert (result.correct, result.total) == (6, 6)
    assert result.unfinished_items == 4
    # Not counted wrong: there is no failed batch and no failed item.
    assert (result.failed_batches, result.failed_items) == (0, 0)
    assert engine.unloaded


def test_finishing_the_last_item_after_the_limit_is_still_too_late() -> None:
    """All five batches judged, the last one ending at 50s against a 45s limit."""
    clock = Clock()
    result = _evaluate(Engine(clock), clock, deadline=45)

    assert result.timed_out is True
    assert result.total == 10
    assert result.unfinished_items == 0


def test_the_load_counts_against_the_limit() -> None:
    clock = Clock()
    engine = Engine(clock, load_seconds=100)
    result = _evaluate(engine, clock, deadline=60)

    assert result.timed_out is True
    assert engine.batches_asked == 0
    assert result.total == 0
    assert result.unfinished_items == 10
    assert engine.unloaded


def test_a_batch_that_fails_in_time_still_counts_wrong() -> None:
    """Failing is the model's answer; running out of time is not."""
    clock = Clock()
    result = _evaluate(Engine(clock, fail_batches=frozenset({1})), clock, deadline=60)

    assert result.timed_out is False
    assert (result.correct, result.total) == (8, 10)
    assert result.failed_items == 2


def test_no_limit_means_no_limit() -> None:
    clock = Clock()
    result = _evaluate(Engine(clock, batch_seconds=10_000), clock, deadline=None)

    assert result.timed_out is False
    assert result.total == 10
