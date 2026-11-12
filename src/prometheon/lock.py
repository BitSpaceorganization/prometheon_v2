"""The cycle lock: one cycle at a time, and never an update in the middle of one.

A cycle holds it for its whole run. An update takes it to switch releases, so
the switch always falls between two cycles. It lives in the validator's state
directory, which every release shares, and is a file lock, so it holds across
processes as well as across the threads of one: the schedule loop runs its
cycle on a thread beside the loop that checks for updates.

A module of its own, with no imports from the CLI, because both the CLI and the
updater take it.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

#: The lock file, inside the state directory.
LOCK_FILE: Final[str] = "cycle.lock"

#: How often a waiting holder looks again at a lock held elsewhere.
_LOCK_POLL_SECONDS: Final[float] = 5.0


@contextmanager
def cycle_lock(
    state_directory: Path,
    *,
    wait: bool = True,
    timeout_seconds: float | None = None,
    poll_seconds: float = _LOCK_POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
) -> Iterator[bool]:
    """Hold the lock a cycle holds while it runs. Yields whether it was taken.

    A cycle takes it for its whole run, and an update takes it to switch, so an
    update never lands in the middle of a cycle. ``wait`` blocks until it is
    free, up to ``timeout_seconds``; without it, a lock held elsewhere yields
    ``False`` at once and nothing waits.

    A separate open file per holder: ``flock`` locks belong to the open file, so
    two holders conflict even inside one process, where the cycle runs on its
    own thread beside the loop that updates.
    """
    state_directory.mkdir(parents=True, exist_ok=True)
    handle = os.open(state_directory / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o644)
    taken = False
    try:
        started = time.monotonic()
        told = False
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                taken = True
                break
            except BlockingIOError:
                if not wait:
                    break
                if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                    break
                if not told:
                    say("a cycle is running; waiting for it to finish")
                    told = True
                sleep(poll_seconds)
        yield taken
    finally:
        if taken:
            fcntl.flock(handle, fcntl.LOCK_UN)
        os.close(handle)


__all__ = ["LOCK_FILE", "cycle_lock"]
