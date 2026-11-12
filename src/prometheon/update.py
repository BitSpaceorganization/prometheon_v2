"""Opt-in updates: a validator moves itself to each new release tag.

Validators update from release tags, never from a branch. With ``[update] auto =
true``, a validator looks at its repository's own ``origin`` every 30 minutes and
moves to the newest release tag ``vX.Y.Z`` that is newer than the version it
runs. Never a branch head, never an older version. ``prometheon update apply``
does the same on demand.

The install is laid out side by side, so moving forward is a symlink switch, and
so is moving back::

    <root>/repo                  a clone of the public repository; tags are fetched here
    <root>/releases/v3.1.0       one checkout per release tag, each with its own .venv
    <root>/releases/v3.2.0
    <root>/current  -> releases/v3.2.0       what runs
    <root>/previous -> releases/v3.1.0       what `prometheon update rollback` returns to

Applying a release, and why the steps come in this order:

1. **Check out the tag beside the others.** The running release is not touched.
2. **``uv sync --frozen``** in it, with the extras this validator needs. The lock
   file decides every version; nothing is resolved anew on a validator.
3. **A self-check, run by the new release itself**: it imports, loads this
   config, finds a hub key pinned, and in score mode runs a CPU canary.
4. **Only then the switch**, while holding the cycle lock, so a cycle is never
   cut in half: ``current`` moves by an atomic rename, and the release it
   pointed at is kept as ``previous``.
5. **The restart**: the systemd units in ``[update] restart_units`` are
   restarted, and a ``validator schedule`` loop re-executes itself from the new
   release.

Anything that fails before the switch leaves ``current`` where it was, and the
reason is logged. A restart that fails after it switches back.

**What is not checked.** The team decided on 2026-11-12 not to verify tag
signatures. A release is trusted because it is a ``vX.Y.Z`` tag on the
repository this install was cloned from.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from prometheon.errors import UpdateError
from prometheon.lock import cycle_lock

#: A release tag: ``v`` and three numbers, without leading zeros. Anything else,
#: a pre-release, a branch, a moved name, is never applied.
_RELEASE_TAG_RE: Final[re.Pattern[str]] = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
_VERSION_RE: Final[re.Pattern[str]] = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)

#: Written into a release directory once it is prepared and checked, so a
#: half-made one is never switched to, and is rebuilt instead of reused.
READY_MARKER: Final[str] = ".prometheon-release-ready"

Version = tuple[int, int, int]


def parse_version(text: str) -> Version:
    """``"3.1.0"`` as ``(3, 1, 0)``. Raises :class:`UpdateError` for anything else."""
    match = _VERSION_RE.match(text.strip())
    if match is None:
        raise UpdateError(f"{text!r} is not a release version (X.Y.Z)")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def release_version(tag: str) -> Version | None:
    """The version a release tag names, or ``None`` for any other name."""
    match = _RELEASE_TAG_RE.match(tag)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def tag_for(version: Version) -> str:
    return "v" + ".".join(str(part) for part in version)


@dataclass(frozen=True)
class Layout:
    """Where an install keeps its clone, its releases and its two links."""

    root: Path

    @property
    def repo(self) -> Path:
        return self.root / "repo"

    @property
    def releases(self) -> Path:
        return self.root / "releases"

    @property
    def current(self) -> Path:
        return self.root / "current"

    @property
    def previous(self) -> Path:
        return self.root / "previous"

    def release(self, tag: str) -> Path:
        return self.releases / tag

    def target(self, link: Path) -> str | None:
        """The release tag a link points at, or ``None`` if it points nowhere."""
        if not link.is_symlink():
            return None
        return Path(os.readlink(link)).name


# --- git --------------------------------------------------------------------------


Runner = Callable[[Sequence[str], Path], str]


def run_command(argv: Sequence[str], cwd: Path) -> str:
    """Run a command and return its output, or raise :class:`UpdateError` saying why."""
    try:
        completed = subprocess.run(  # noqa: S603 - a fixed argv, never a shell
            list(argv), cwd=cwd, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise UpdateError(f"could not run {argv[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise UpdateError(
            f"{' '.join(argv[:3])} failed ({completed.returncode}): "
            f"{detail[-1] if detail else 'no output'}"
        )
    return completed.stdout


def release_tags(layout: Layout, *, run: Runner = run_command) -> dict[Version, str]:
    """Every release tag the local clone knows, by version."""
    output = run(["git", "tag", "--list", "v*"], layout.repo)
    tags: dict[Version, str] = {}
    for line in output.splitlines():
        name = line.strip()
        version = release_version(name)
        if version is not None:
            tags[version] = name
    return tags


def fetch_tags(layout: Layout, *, run: Runner = run_command) -> None:
    """Fetch tags from the clone's own ``origin``; never from anywhere else."""
    run(["git", "fetch", "--quiet", "--tags", "--force", "origin"], layout.repo)


def tag_commit(layout: Layout, tag: str, *, run: Runner = run_command) -> str:
    """The commit a release tag points at, read from the tag, never a branch."""
    return run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], layout.repo).strip()


def newest_release(available: dict[Version, str], running: Version) -> tuple[Version, str] | None:
    """The newest release strictly newer than ``running``, or ``None``."""
    newer = [version for version in available if version > running]
    if not newer:
        return None
    best = max(newer)
    return best, available[best]


# --- the release directories --------------------------------------------------------


def switch(layout: Layout, tag: str) -> str | None:
    """Point ``current`` at ``releases/<tag>``, keep the old target as ``previous``.

    Each link moves by writing a new one beside it and renaming it over the old,
    which is atomic: a process starting at any instant finds a whole link,
    never a missing one. Returns the tag ``current`` pointed at before.
    """
    before = layout.target(layout.current)
    _relink(layout.current, Path("releases") / tag)
    if before is not None and before != tag:
        _relink(layout.previous, Path("releases") / before)
    return before


def _relink(link: Path, target: Path) -> None:
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


Prepare = Callable[[Path, Sequence[str]], None]
SelfCheck = Callable[[Path], None]
Restart = Callable[[], None]


def uv_sync(release: Path, extras: Sequence[str]) -> None:
    """``uv sync --frozen`` with the extras the validator needs, inside one release.

    ``--frozen``: the release's own lock file decides every version, and is
    never re-resolved on a validator. ``--no-dev``: a validator runs no tests.
    """
    uv = os.environ.get("UV") or shutil.which("uv")
    if uv is None:
        raise UpdateError("uv is not on PATH, so a release cannot be installed")
    argv = [uv, "sync", "--frozen", "--no-dev"]
    for extra in extras:
        argv += ["--extra", extra]
    run_command(argv, release)


@dataclass(frozen=True)
class Outcome:
    """What an update attempt did, for the log and for the caller."""

    #: ``applied``, ``current`` (nothing newer), ``busy`` (a cycle held the
    #: lock and the caller would not wait) or ``failed``.
    status: str
    running: Version
    tag: str | None = None
    detail: str = ""


def install(
    layout: Layout,
    tag: str,
    *,
    extras: Sequence[str],
    prepare: Prepare,
    self_check: SelfCheck,
    run: Runner = run_command,
) -> Path:
    """Check out, prepare and check one release beside the others. Returns its path.

    A release marked ready is reused, but only if it still holds the commit its
    tag names: a tag moved after install is refused rather than trusted twice.
    A release that fails to prepare or check is removed, so the next attempt
    starts clean and nothing half-made is ever switched to.
    """
    release = layout.release(tag)
    commit = tag_commit(layout, tag, run=run)
    if (release / READY_MARKER).is_file():
        held = (release / READY_MARKER).read_text(encoding="utf-8").strip()
        if held != commit:
            raise UpdateError(
                f"{tag} points at {commit[:12]} now, but {release} was installed from "
                f"{held[:12]}: the tag moved after it was installed, so it is not used"
            )
        return release
    _remove_release(layout, release, run=run)
    layout.releases.mkdir(parents=True, exist_ok=True)
    # The commit, not the name: a branch named like a release can never be
    # what is checked out.
    run(["git", "worktree", "add", "--force", "--detach", str(release), commit], layout.repo)
    try:
        prepare(release, extras)
        self_check(release)
    except BaseException:
        _remove_release(layout, release, run=run)
        raise
    (release / READY_MARKER).write_text(commit + "\n", encoding="utf-8")
    return release


def _remove_release(layout: Layout, release: Path, *, run: Runner) -> None:
    """Remove a release directory that nothing points at."""
    for link in (layout.current, layout.previous):
        if layout.target(link) == release.name:
            raise UpdateError(f"refusing to remove {release}: {link.name} points at it")
    if not release.exists():
        return
    try:
        run(["git", "worktree", "remove", "--force", str(release)], layout.repo)
    except UpdateError:
        shutil.rmtree(release, ignore_errors=True)
        run(["git", "worktree", "prune"], layout.repo)


def apply(
    layout: Layout,
    *,
    running: Version,
    state_directory: Path,
    extras: Sequence[str],
    self_check: SelfCheck,
    restart: Restart,
    prepare: Prepare = uv_sync,
    wait: bool = True,
    lock_timeout_seconds: float | None = None,
    fetch: bool = True,
    run: Runner = run_command,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
) -> Outcome:
    """Move to the newest release tag newer than ``running``, or do nothing.

    Never raises for a release that cannot be installed: the outcome says
    ``failed`` and why, and ``current`` has not moved. Raises
    :class:`UpdateError` only when the install itself is not one this can
    update: no clone to fetch tags into.
    """
    if not (layout.repo / ".git").exists():
        raise UpdateError(
            f"no clone at {layout.repo}. Lay the install out as docs/validator.md describes "
            "before turning on updates"
        )
    if fetch:
        try:
            fetch_tags(layout, run=run)
        except UpdateError as exc:
            say(f"could not fetch release tags, staying on {tag_for(running)}: {exc}")
            return Outcome("failed", running, None, str(exc))
    found = newest_release(release_tags(layout, run=run), running)
    if found is None:
        return Outcome("current", running)
    _version, tag = found
    say(f"{tag} is newer than the running {tag_for(running)}; installing it beside it")
    try:
        install(layout, tag, extras=extras, prepare=prepare, self_check=self_check, run=run)
    except UpdateError as exc:
        say(f"staying on {tag_for(running)}: {tag} did not install: {exc}")
        return Outcome("failed", running, tag, str(exc))

    with cycle_lock(
        state_directory,
        wait=wait,
        timeout_seconds=lock_timeout_seconds,
        sleep=sleep,
        say=say,
    ) as taken:
        if not taken:
            say(f"{tag} is ready; a cycle is running, so it is switched to on a later check")
            return Outcome("busy", running, tag, "a cycle holds the lock")
        before = switch(layout, tag)
        try:
            restart()
        except Exception as exc:
            if before is not None:
                switch(layout, before)
            say(f"restarting on {tag} failed, so {before} is current again: {exc}")
            return Outcome("failed", running, tag, f"restart failed: {exc}")
    say(f"updated from {tag_for(running)} to {tag}")
    return Outcome("applied", running, tag)


def rollback(
    layout: Layout,
    *,
    state_directory: Path,
    restart: Restart,
    wait: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    say: Callable[[str], None] = lambda _message: None,
) -> str:
    """Switch ``current`` back to ``previous``. Returns the tag now current.

    Rollback is the one move to an older release, and only ever to the one
    that ran before, kept installed for exactly this.
    """
    current = layout.target(layout.current)
    previous = layout.target(layout.previous)
    # The directory, not the ready marker: the release an operator started on
    # was installed by hand and never marked, and is exactly what to go back to.
    if previous is None or not layout.release(previous).is_dir():
        raise UpdateError("there is no previous release to roll back to")
    with cycle_lock(state_directory, wait=wait, sleep=sleep, say=say) as taken:
        if not taken:
            raise UpdateError("a cycle is running; roll back once it has finished")
        _relink(layout.current, Path("releases") / previous)
        if current is not None:
            _relink(layout.previous, Path("releases") / current)
        restart()
    say(f"rolled back from {current} to {previous}")
    return previous


def restart_units(units: Sequence[str], *, run: Runner = run_command) -> None:
    """``systemctl try-restart`` each unit: running ones restart, stopped ones stay.

    The shipped units are timers that start one-shot services from
    ``<root>/current``, so the next run is the new release and nothing needs
    restarting. A long-running process, such as ``validator schedule`` under
    its own unit, is named in ``[update] restart_units``.
    """
    for unit in units:
        run(["systemctl", "try-restart", unit], Path("/"))


def installed_layout(package_file: str) -> Layout | None:
    """The layout the running code belongs to, or ``None`` outside one.

    The code runs from ``<root>/releases/<tag>/src/prometheon``, reached through
    ``<root>/current``; resolving the link finds the release, and its parent's
    parent is the root.
    """
    project = Path(package_file).resolve().parents[2]
    if project.parent.name != "releases" or release_version(project.name) is None:
        return None
    return Layout(root=project.parent.parent)


__all__ = [
    "READY_MARKER",
    "Layout",
    "Outcome",
    "apply",
    "fetch_tags",
    "install",
    "installed_layout",
    "newest_release",
    "parse_version",
    "release_tags",
    "release_version",
    "restart_units",
    "rollback",
    "run_command",
    "switch",
    "tag_commit",
    "tag_for",
    "uv_sync",
]
