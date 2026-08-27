"""The clock a validator runs on, in-process.

Two jobs on two cadences: the daily cycle that scores yesterday, and the
re-post that keeps the resulting vector counting between cycles. Cron can do
both, and the guide shows how; this exists because a validator host frequently
has neither cron nor systemd, and because a relative ``sleep`` gets the second
job subtly wrong.

**Wake-ups are aligned to the wall clock, not to when the process started.**
A loop that sleeps 1800 seconds from launch drifts to wherever it happened to
be started, so a field of validators ends up scattered uniformly across the
half hour. Sleeping to the next :00/:30 boundary instead puts every validator
running this on the same instants, whenever each was started and however long
its last cycle took. That is worth having: validators submitting together are
compared against the same recent chain state, and an operator debugging a
divergence can line two logs up by timestamp instead of by guesswork.

Alignment is not a consensus requirement -- Yuma compares whatever is on chain
within a tempo, and nothing breaks if a validator is late. It is a coordination
convenience, and it costs nothing to have.

**The two jobs never overlap**, because they are one loop rather than two
timers. A cycle that runs long simply delays the next re-post to the following
boundary; it cannot submit weights underneath itself. The shell equivalent
needs a lock file to get that right, and this does not.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable

#: Re-post boundaries. Thirty minutes sits between `weights_set_rate_limit`
#: (100 blocks, ~20 min: submitting faster is refused) and `activity_cutoff`
#: (720 blocks, ~2.4 h: submitting slower stops counting), with about five
#: posts per cutoff window so losing one to a collision is free.
RESUBMIT_MINUTES: tuple[int, ...] = (0, 30)

#: When the daily cycle starts, UTC. Late enough that the DB layer has finished
#: building the previous day -- it publishes days before it has finished them,
#: so starting at 00:05 fetches a snapshot that is still growing.
DEFAULT_CYCLE_HOUR = 4


def seconds_until_next_boundary(
    now: dt.datetime, minutes: tuple[int, ...] = RESUBMIT_MINUTES
) -> float:
    """Seconds from ``now`` to the next wall-clock boundary in ``minutes``.

    Never returns 0: landing exactly on a boundary sleeps a full interval
    rather than spinning. Computed from the wall clock so two processes started
    minutes apart converge on the same schedule instead of staying offset.
    """
    if not minutes:
        raise ValueError("at least one boundary minute is required")
    for candidate in sorted(minutes):
        target = now.replace(minute=candidate, second=0, microsecond=0)
        if target > now:
            return (target - now).total_seconds()
    # Past the last boundary this hour: the first one of the next.
    nxt = (now + dt.timedelta(hours=1)).replace(minute=min(minutes), second=0, microsecond=0)
    return (nxt - now).total_seconds()


def is_cycle_due(now: dt.datetime, *, cycle_hour: int, last_cycle_day: dt.date | None) -> bool:
    """Whether the daily cycle should run at ``now``.

    Due once per UTC day, at or after ``cycle_hour``. ``last_cycle_day`` is the
    UTC date of the last attempt, so a restart mid-morning still runs the day
    it missed rather than waiting for tomorrow -- the day it skipped is the one
    miners are waiting to be paid for.
    """
    if now.hour < cycle_hour:
        return False
    return last_cycle_day != now.date()


def run_forever(
    *,
    cycle: Callable[[], None],
    resubmit: Callable[[], None],
    cycle_hour: int = DEFAULT_CYCLE_HOUR,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc),
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
    max_iterations: int | None = None,
) -> None:
    """Run the cycle daily and the re-post on every boundary, indefinitely.

    ``cycle`` and ``resubmit`` are injected rather than imported so this module
    holds no chain or database dependency and a test can drive a year of
    schedule in milliseconds. Both are called for their effect; a raising job is
    reported and the loop continues, because one failed cycle must not stop the
    re-posts that keep yesterday's vector alive.
    """
    last_cycle_day: dt.date | None = None
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        current = now()
        if is_cycle_due(current, cycle_hour=cycle_hour, last_cycle_day=last_cycle_day):
            # Marked before running, not after: a cycle that raises half way
            # through has still consumed its slot for the day, and retrying it
            # every 30 minutes would re-label the same corpus repeatedly.
            last_cycle_day = current.date()
            say(f"cycle due for {current.date().isoformat()}")
            try:
                cycle()
            except Exception as exc:
                # Reported, never fatal: one failed cycle must not stop the
                # re-posts that keep yesterday's vector counting.
                say(f"cycle failed: {exc}")
        else:
            try:
                resubmit()
            except Exception as exc:
                # A transient chain or DB error must not end the loop; the next
                # boundary is 30 minutes away and well inside activity_cutoff.
                say(f"re-post failed: {exc}")

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            return
        sleep(seconds_until_next_boundary(now()))


__all__ = [
    "DEFAULT_CYCLE_HOUR",
    "RESUBMIT_MINUTES",
    "is_cycle_due",
    "run_forever",
    "seconds_until_next_boundary",
]
