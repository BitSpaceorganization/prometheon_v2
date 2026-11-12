"""The validator's in-process clock.

The property under test throughout is *alignment*: two validators started at
different moments must converge on the same wake-up instants. A relative sleep
does not do that, and the failure is invisible -- everything works, the field is
just smeared across the half hour.
"""

from __future__ import annotations

import argparse
import datetime as dt
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from prometheon.cli.schedule import (
    MAX_CYCLE_ATTEMPTS,
    RESUBMIT_MINUTES,
    is_cycle_due,
    run_forever,
    seconds_until_next_boundary,
    start_thread,
)
from prometheon.errors import EpochNotReadyError

pytestmark = pytest.mark.unit


def _at(hour: int, minute: int, second: int = 0) -> dt.datetime:
    return dt.datetime(2026, 8, 27, hour, minute, second, tzinfo=dt.timezone.utc)


class TestBoundaryAlignment:
    @pytest.mark.parametrize(
        ("now", "expected"),
        [
            (_at(4, 0, 1), 29 * 60 + 59),  # just past :00 -> :30
            (_at(4, 29, 0), 60),  # a minute short of :30
            (_at(4, 31, 0), 29 * 60),  # just past :30 -> next hour's :00
            (_at(4, 59, 30), 30),  # -> 05:00
        ],
    )
    def test_it_sleeps_to_the_wall_clock_not_a_fixed_interval(
        self, now: dt.datetime, expected: float
    ) -> None:
        assert seconds_until_next_boundary(now) == expected

    def test_two_validators_started_apart_converge_on_the_same_instants(self) -> None:
        """The reason this module exists.

        `sleep(1800)` from launch keeps whatever offset the process started
        with, forever. Sleeping to the boundary erases it after one wake-up.
        """

        def landing(t: dt.datetime) -> dt.datetime:
            return t + dt.timedelta(seconds=seconds_until_next_boundary(t))

        a, b = _at(4, 3, 17), _at(4, 21, 49)
        assert landing(a) == landing(b) == _at(4, 30)

    def test_landing_exactly_on_a_boundary_waits_a_full_interval(self) -> None:
        """Not zero, which would spin and re-post in a tight loop."""
        assert seconds_until_next_boundary(_at(4, 30, 0)) == 30 * 60

    def test_the_boundaries_are_the_documented_ones(self) -> None:
        """30 minutes: above the ~20 min rate limit, well under the 2.4 h cutoff."""
        assert RESUBMIT_MINUTES == (0, 30)


class TestWhenTheCycleIsDue:
    """Due means: at or after the hour, and yesterday is not finished yet."""

    def test_not_before_the_configured_hour(self) -> None:
        assert not is_cycle_due(_at(3, 59), cycle_hour=4, last_finished_day=None)

    def test_due_at_the_hour_when_nothing_has_finished(self) -> None:
        assert is_cycle_due(_at(4, 0), cycle_hour=4, last_finished_day=None)

    def test_not_due_once_yesterday_is_finished(self) -> None:
        """B2: the state file holds the day *scored*, which is yesterday.

        v2 compared that with today, so every restart re-ran a finished cycle.
        """
        yesterday = _at(0, 0).date() - dt.timedelta(days=1)
        assert not is_cycle_due(_at(9, 30), cycle_hour=4, last_finished_day=yesterday)

    def test_due_when_the_last_finished_day_is_older(self) -> None:
        two_days_ago = _at(0, 0).date() - dt.timedelta(days=2)
        assert is_cycle_due(_at(11, 0), cycle_hour=4, last_finished_day=two_days_ago)


def _clock(start: dt.datetime) -> tuple[Any, Any]:
    state = {"now": start}
    return (
        (lambda: state["now"]),
        (lambda seconds: state.__setitem__("now", state["now"] + dt.timedelta(seconds=seconds))),
    )


@dataclass
class _Done:
    """A cycle that ran inline: finished at once, with or without an error."""

    day: dt.date
    _error: BaseException | None = None

    def finished(self) -> bool:
        return True

    def error(self) -> BaseException | None:
        return self._error


def _inline(cycle: Any, day: dt.date) -> _Done:
    try:
        cycle(day)
    except Exception as exc:
        return _Done(day, exc)
    return _Done(day)


YESTERDAY = _at(0, 0).date() - dt.timedelta(days=1)


class TestTheLoop:
    def test_it_scores_yesterday_once_and_reposts_on_every_boundary(self) -> None:
        now, sleep = _clock(_at(3, 0))
        cycles: list[dt.date] = []
        reposts: list[dt.datetime] = []
        run_forever(
            cycle=cycles.append,
            resubmit=lambda: reposts.append(now()),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            max_iterations=8,
        )
        assert cycles == [YESTERDAY], "exactly one cycle, for yesterday"
        assert len(reposts) == 8, "every wake-up re-posts, including the one that starts a cycle"
        assert all(r.minute in RESUBMIT_MINUTES and r.second == 0 for r in reposts[1:])

    def test_a_failed_cycle_is_retried_a_bounded_number_of_times(self) -> None:
        """B20: one transient failure no longer costs the day, but a real fault
        is not retried forever either."""
        now, sleep = _clock(_at(4, 0))
        attempts: list[dt.date] = []

        def boom(day: dt.date) -> None:
            attempts.append(day)
            raise RuntimeError("labelling endpoint down")

        run_forever(
            cycle=boom,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            max_iterations=12,
        )
        assert attempts == [YESTERDAY] * MAX_CYCLE_ATTEMPTS

    def test_an_epoch_not_built_yet_is_retried_without_counting(self) -> None:
        """B20: a validator that starts before the hub has built the day waits."""
        now, sleep = _clock(_at(4, 0))
        attempts: list[dt.date] = []

        def not_yet(day: dt.date) -> None:
            attempts.append(day)
            if len(attempts) < 6:
                raise EpochNotReadyError("409")

        run_forever(
            cycle=not_yet,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            max_iterations=12,
        )
        assert len(attempts) == 6, "five not-ready answers did not use up the attempts"

    def test_a_failing_cycle_does_not_stop_the_reposts(self) -> None:
        now, sleep = _clock(_at(3, 59))
        reposts: list[dt.datetime] = []

        def boom(_day: dt.date) -> None:
            raise RuntimeError("down")

        run_forever(
            cycle=boom,
            resubmit=lambda: reposts.append(now()),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            max_iterations=5,
        )
        assert len(reposts) == 5


class TestRepostsContinueWhileACycleRuns:
    def test_a_long_cycle_on_its_thread_does_not_silence_the_reposts(self) -> None:
        """B27: v2 ran both jobs in one loop, so a cycle of many hours masked the
        validator out of consensus while it computed. Uses a real thread."""
        now, sleep_clock = _clock(_at(4, 0))
        release = threading.Event()
        started = threading.Event()
        reposts: list[dt.datetime] = []

        def long_cycle(_day: dt.date) -> None:
            started.set()
            assert release.wait(timeout=10)

        def sleep(seconds: float) -> None:
            sleep_clock(seconds)

        def repost() -> None:
            reposts.append(now())
            if len(reposts) == 4:
                release.set()

        run_forever(
            cycle=long_cycle,
            resubmit=repost,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=start_thread,
            max_iterations=4,
        )
        assert started.is_set()
        assert len(reposts) == 4, "re-posts kept their schedule while the cycle ran"


class TestTheUpdateCheck:
    """With [update] auto on, the loop looks for a release on every boundary."""

    def test_it_runs_on_every_boundary_while_no_cycle_runs(self) -> None:
        now, sleep = _clock(_at(1, 0))
        checks: list[dt.datetime] = []
        run_forever(
            cycle=lambda _day: None,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            max_iterations=3,
            update=lambda: checks.append(now()),
        )
        assert len(checks) == 3

    def test_it_never_runs_while_a_cycle_runs_here(self) -> None:
        """An update must not land in the middle of a cycle."""

        @dataclass
        class Running:
            day: dt.date

            def finished(self) -> bool:
                return False

            def error(self) -> BaseException | None:
                return None

        now, sleep = _clock(_at(4, 0))
        checks: list[int] = []
        run_forever(
            cycle=lambda _day: None,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=lambda _cycle, day: Running(day),
            max_iterations=3,
            update=lambda: checks.append(1),
        )
        assert checks == []

    def test_a_failing_check_stops_neither_the_loop_nor_the_reposts(self) -> None:
        now, sleep = _clock(_at(1, 0))
        reposts: list[int] = []
        said: list[str] = []

        def broken() -> None:
            raise RuntimeError("origin unreachable")

        run_forever(
            cycle=lambda _day: None,
            resubmit=lambda: reposts.append(1),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            say=said.append,
            max_iterations=2,
            update=broken,
        )
        assert len(reposts) == 2
        assert any("update check failed" in line for line in said)


class TestSurvivingARestart:
    def test_a_restart_does_not_rerun_a_finished_day(self) -> None:
        now, sleep = _clock(_at(9, 0))
        cycles: list[dt.date] = []
        run_forever(
            cycle=cycles.append,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            last_finished_day=lambda: YESTERDAY,  # what the state file says
            max_iterations=6,
        )
        assert cycles == []

    def test_a_restart_runs_a_day_that_was_never_finished(self) -> None:
        now, sleep = _clock(_at(9, 0))
        cycles: list[dt.date] = []
        run_forever(
            cycle=cycles.append,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            spawn=_inline,
            last_finished_day=lambda: YESTERDAY - dt.timedelta(days=1),
            max_iterations=4,
        )
        assert cycles == [YESTERDAY]


class TestTheScheduleCommand:
    def test_a_scheduled_local_cycle_is_given_its_day_and_policy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """B1: v2 built the cycle's arguments without `policy`, so every scheduled
        local cycle died on an AttributeError and nothing was ever scored."""
        from prometheon.cli import validator as cli

        config = SimpleNamespace(
            validator=SimpleNamespace(
                mode=SimpleNamespace(value="score"),
                state_directory=tmp_path,
                submit_weights=True,
                submit_results=True,
            ),
            update=SimpleNamespace(auto=False),
        )
        seen: list[argparse.Namespace] = []
        monkeypatch.setattr(cli, "load", lambda _path: config)
        monkeypatch.setattr(cli, "cmd_run", lambda args: seen.append(args) or 0)
        monkeypatch.setattr(
            "prometheon.cli.schedule.start_thread", lambda cycle, day: _inline(cycle, day)
        )

        def fake_run_forever(**kwargs: Any) -> None:
            kwargs["cycle"](YESTERDAY)

        monkeypatch.setattr("prometheon.cli.schedule.run_forever", fake_run_forever)
        cli.cmd_schedule(
            argparse.Namespace(
                config="cfg.toml", cycle_hour=4, max_iterations=1, policy="policy.md"
            )
        )
        (args,) = seen
        assert args.policy == "policy.md"
        assert args.date == YESTERDAY.isoformat()
