"""The validator's in-process clock.

The property under test throughout is *alignment*: two validators started at
different moments must converge on the same wake-up instants. A relative sleep
does not do that, and the failure is invisible -- everything works, the field is
just smeared across the half hour.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prometheon.cli.schedule import (
    RESUBMIT_MINUTES,
    is_cycle_due,
    run_forever,
    seconds_until_next_boundary,
)

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
    def test_not_before_the_configured_hour(self) -> None:
        assert not is_cycle_due(_at(3, 59), cycle_hour=4, last_cycle_day=None)

    def test_due_at_the_hour_when_it_has_not_run_today(self) -> None:
        assert is_cycle_due(_at(4, 0), cycle_hour=4, last_cycle_day=None)

    def test_not_due_twice_in_one_day(self) -> None:
        assert not is_cycle_due(_at(9, 30), cycle_hour=4, last_cycle_day=_at(4, 0).date())

    def test_a_restart_after_the_hour_still_runs_the_day_it_missed(self) -> None:
        """The skipped day is the one miners are waiting to be paid for."""
        assert is_cycle_due(_at(11, 0), cycle_hour=4, last_cycle_day=dt.date(2026, 8, 26))


class TestTheLoop:
    def _clock(self, start: dt.datetime):
        state = {"now": start}
        return (
            (lambda: state["now"]),
            (lambda s: state.__setitem__("now", state["now"] + dt.timedelta(seconds=s))),
        )

    def test_it_reposts_on_boundaries_and_cycles_once_a_day(self) -> None:
        now, sleep = self._clock(_at(3, 0))
        cycles: list[dt.date] = []
        reposts: list[dt.datetime] = []
        run_forever(
            cycle=lambda: cycles.append(now().date()),
            resubmit=lambda: reposts.append(now()),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            max_iterations=8,
        )
        assert cycles == [dt.date(2026, 8, 27)], "exactly one cycle for the day"
        # Everything else is a re-post, and every one lands on a boundary.
        assert reposts, "the loop must keep re-posting between cycles"
        assert all(r.minute in RESUBMIT_MINUTES and r.second == 0 for r in reposts)

    def test_a_failing_cycle_does_not_stop_the_reposts(self) -> None:
        """Yesterday's vector still needs to keep counting while today is broken."""
        now, sleep = self._clock(_at(3, 59))
        reposts = []

        def boom() -> None:
            raise RuntimeError("labelling endpoint down")

        run_forever(
            cycle=boom,
            resubmit=lambda: reposts.append(now()),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            max_iterations=5,
        )
        assert len(reposts) >= 3

    def test_a_failing_cycle_is_not_retried_every_boundary(self) -> None:
        """Retrying would re-label the same corpus every 30 minutes, and
        labelling is the only real bill."""
        now, sleep = self._clock(_at(4, 0))
        attempts = []

        def boom() -> None:
            attempts.append(now())
            raise RuntimeError("nope")

        run_forever(
            cycle=boom,
            resubmit=lambda: None,
            cycle_hour=4,
            now=now,
            sleep=sleep,
            max_iterations=10,
        )
        assert len(attempts) == 1

    def test_the_two_jobs_never_overlap(self) -> None:
        """One loop, not two timers: a long cycle delays the next re-post
        rather than submitting weights underneath itself."""
        now, sleep = self._clock(_at(4, 0))
        running = {"cycle": False}
        overlaps = []

        def slow_cycle() -> None:
            running["cycle"] = True
            sleep(3 * 3600)  # a cycle that runs three hours
            running["cycle"] = False

        run_forever(
            cycle=slow_cycle,
            resubmit=lambda: overlaps.append(running["cycle"]),
            cycle_hour=4,
            now=now,
            sleep=sleep,
            max_iterations=6,
        )
        assert not any(overlaps), "a re-post ran while the cycle was in flight"
