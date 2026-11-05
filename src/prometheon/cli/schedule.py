"""The clock a validator runs on, in-process.

Two jobs on two cadences: the daily cycle that scores yesterday, and the
re-post that keeps the last vector counting between cycles. Cron or systemd
can do both; this exists for a host that has neither.

**Wake-ups are aligned to the wall clock, not to when the process started.**
Sleeping to the next :00/:30 boundary puts every validator running this on the
same instants, whenever each was started and however long its last cycle took.

**The cycle runs on its own thread, and re-posts never stop.** A local cycle
labels and evaluates for hours, while weights stop counting toward consensus
once ``activity_cutoff`` passes. v2 ran both jobs in one sequential loop, so a
validator's row went silent for most of every day while it computed (B27). Now
the loop keeps re-posting on every boundary while the cycle works. The two
cannot submit underneath each other: the cycle saves its new vector before it
sends, so a re-post after that sends the new vector, and the chain's rate limit
refuses whichever of two sends lands second.

**A day is due until it is finished.** Due means: at or after ``cycle_hour``,
and the last *finished* day is older than yesterday. v2 compared the day the
state file had scored with today, so every restart re-ran a cycle that had
already finished (B2). A cycle that failed is retried at the next boundary,
up to ``max_attempts`` times a day, and a cycle that found the hub's epoch not
built yet is retried without counting: a validator started at 04:00 when the
hub was late used to lose the whole day (B20).
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol

from prometheon.errors import EpochNotReadyError

#: Re-post boundaries. Thirty minutes sits between the weights rate limit
#: (submitting faster is refused) and ``activity_cutoff`` (submitting slower
#: stops counting), with room to lose one to a collision.
RESUBMIT_MINUTES: tuple[int, ...] = (0, 30)

#: When the daily cycle starts, UTC. The hub builds each day's epoch at 01:00
#: and retries BitFan's feed until 03:30, so by 04:00 the epoch is normally
#: there; a cycle that still finds it missing waits for the next boundary.
DEFAULT_CYCLE_HOUR = 4

#: Failed attempts at one day before the scheduler leaves it to a person.
#: Resuming is cheap, because a computed day is saved, so a few retries cost
#: little; unlimited retries would hide a fault that needs fixing.
MAX_CYCLE_ATTEMPTS: Final[int] = 3


def seconds_until_next_boundary(
    now: dt.datetime, minutes: tuple[int, ...] = RESUBMIT_MINUTES
) -> float:
    """Seconds from ``now`` to the next wall-clock boundary in ``minutes``.

    Never returns 0: landing exactly on a boundary sleeps a full interval
    rather than spinning.
    """
    if not minutes:
        raise ValueError("at least one boundary minute is required")
    for candidate in sorted(minutes):
        target = now.replace(minute=candidate, second=0, microsecond=0)
        if target > now:
            return (target - now).total_seconds()
    nxt = (now + dt.timedelta(hours=1)).replace(minute=min(minutes), second=0, microsecond=0)
    return (nxt - now).total_seconds()


def is_cycle_due(now: dt.datetime, *, cycle_hour: int, last_finished_day: dt.date | None) -> bool:
    """Whether yesterday's cycle should run at ``now``.

    ``last_finished_day`` is the last day a cycle *scored and finished*, not the
    date it ran on: a cycle that ran on the 5th scored the 4th.
    """
    if now.hour < cycle_hour:
        return False
    target = now.date() - dt.timedelta(days=1)
    return last_finished_day is None or last_finished_day < target


class Job(Protocol):
    day: dt.date

    def finished(self) -> bool: ...

    def error(self) -> BaseException | None: ...


@dataclass
class ThreadJob:
    """One cycle running on a daemon thread."""

    day: dt.date
    _thread: threading.Thread | None = None
    _error: BaseException | None = None
    _done: threading.Event = field(default_factory=threading.Event)

    def finished(self) -> bool:
        return self._done.is_set()

    def error(self) -> BaseException | None:
        return self._error


def start_thread(cycle: Callable[[dt.date], None], day: dt.date) -> Job:
    job = ThreadJob(day=day)

    def target() -> None:
        try:
            cycle(day)
        except BaseException as exc:  # reported by the loop, never lost
            job._error = exc
        finally:
            job._done.set()

    job._thread = threading.Thread(target=target, name=f"cycle-{day.isoformat()}", daemon=True)
    job._thread.start()
    return job


def run_forever(
    *,
    cycle: Callable[[dt.date], None],
    resubmit: Callable[[], None],
    cycle_hour: int = DEFAULT_CYCLE_HOUR,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
    last_finished_day: Callable[[], dt.date | None] = lambda: None,
    max_iterations: int | None = None,
    spawn: Callable[[Callable[[dt.date], None], dt.date], Job] = start_thread,
    max_attempts: int = MAX_CYCLE_ATTEMPTS,
) -> None:
    """Start each day's cycle when due, and re-post on every boundary, indefinitely.

    ``cycle`` receives the day to score. ``last_finished_day`` is read from the
    validator's state on every wake-up, so a restart sees what the last process
    finished. ``spawn`` runs a cycle; the default is a daemon thread, and a test
    can pass one that runs it inline.
    """
    job: Job | None = None
    attempts: dict[dt.date, int] = {}
    finished_here: dt.date | None = None
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        current = now()
        if job is not None and job.finished():
            error = job.error()
            if error is None:
                finished_here = job.day
                say(f"cycle for {job.day.isoformat()} finished")
            elif isinstance(error, EpochNotReadyError):
                say(f"the epoch for {job.day.isoformat()} is not built yet; trying again shortly")
            else:
                attempts[job.day] = attempts.get(job.day, 0) + 1
                say(
                    f"cycle for {job.day.isoformat()} failed "
                    f"({attempts[job.day]}/{max_attempts}): {error}"
                )
            job = None

        target = current.date() - dt.timedelta(days=1)
        finished = max(
            (day for day in (last_finished_day(), finished_here) if day is not None),
            default=None,
        )
        if (
            job is None
            and is_cycle_due(current, cycle_hour=cycle_hour, last_finished_day=finished)
            and attempts.get(target, 0) < max_attempts
        ):
            say(f"cycle due for {target.isoformat()}")
            job = spawn(cycle, target)

        try:
            resubmit()
        except Exception as exc:
            # A transient chain or hub error must not end the loop; the next
            # boundary is thirty minutes away and well inside activity_cutoff.
            say(f"re-post failed: {exc}")

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return
        sleep(seconds_until_next_boundary(now()))


__all__ = [
    "DEFAULT_CYCLE_HOUR",
    "MAX_CYCLE_ATTEMPTS",
    "RESUBMIT_MINUTES",
    "ThreadJob",
    "is_cycle_due",
    "run_forever",
    "seconds_until_next_boundary",
    "start_thread",
]
