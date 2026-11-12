"""Updates: release tags only, newer only, checked before the switch, never mid-cycle.

Git is real here: an origin repository with release tags, a clone, and release
worktrees beside each other, exactly as an install lays them out. What is
injected is what would install packages or start processes: ``uv sync``, the
new release's self-check, and the restart. Each test drives the same
:func:`prometheon.update.apply` the CLI and the schedule loop call.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from prometheon import update as updater
from prometheon.errors import UpdateError
from prometheon.lock import cycle_lock

from .conftest import World, git

pytestmark = pytest.mark.unit


# --- what is applied -------------------------------------------------------------


def test_a_newer_release_tag_is_installed_checked_and_switched_to(world: World) -> None:
    world.release("3.2.0")

    outcome = world.apply("3.1.0")

    assert (outcome.status, outcome.tag) == ("applied", "v3.2.0")
    assert (world.current(), world.previous()) == ("v3.2.0", "v3.1.0")
    assert ("v3.2.0", ("wrapper",)) in world.prepared
    assert "v3.2.0" in world.checked
    assert world.restarts == 1
    # The release holds exactly the tagged commit, checked out beside the old one.
    release = world.layout.release("v3.2.0")
    assert (release / "VERSION").read_text() == "3.2.0\n"
    assert (world.layout.release("v3.1.0") / "VERSION").read_text() == "3.1.0\n"


def test_the_newest_of_several_newer_releases_is_the_one_applied(world: World) -> None:
    world.release("3.1.1")
    world.release("3.2.0")
    world.release("3.10.0")

    assert world.apply("3.1.0").tag == "v3.10.0"


def test_nothing_older_is_ever_applied(world: World) -> None:
    """No downgrade: the running version is newer than every tag."""
    world.release("3.2.0")

    outcome = world.apply("3.3.0")

    assert outcome.status == "current"
    assert world.current() == "v3.1.0"
    assert world.restarts == 0


def test_the_same_version_is_not_reapplied(world: World) -> None:
    assert world.apply("3.1.0").status == "current"


@pytest.mark.parametrize("name", ["v3.3.0-rc1", "v3.3", "3.3.0", "release-3.3.0", "v03.3.0"])
def test_only_a_release_tag_counts(world: World, name: str) -> None:
    world.release("3.3.0", tag=name)

    assert world.apply("3.1.0").status == "current"
    assert world.current() == "v3.1.0"


def test_a_branch_head_is_never_applied(world: World) -> None:
    """A branch named like a release is not a release."""
    (world.work / "VERSION").write_text("9.9.9\n")
    git("checkout", "--quiet", "-b", "v9.9.9", cwd=world.work)
    git("commit", "--quiet", "-am", "not a release", cwd=world.work)
    git("push", "--quiet", "origin", "v9.9.9", cwd=world.work)

    assert world.apply("3.1.0").status == "current"


def test_tags_come_from_the_installs_own_origin(world: World) -> None:
    world.release("3.2.0")
    remotes = git("remote", "-v", cwd=world.layout.repo)

    world.apply("3.1.0")

    assert str(world.origin) in remotes
    assert git("tag", "--list", cwd=world.layout.repo).split() == ["v3.1.0", "v3.2.0"]


# --- what a failure leaves --------------------------------------------------------


def test_a_failed_self_check_leaves_the_current_release_running(world: World) -> None:
    world.release("3.2.0")

    def broken(release: Path) -> None:
        raise UpdateError("the new release cannot load this config")

    outcome = world.apply("3.1.0", self_check=broken)

    assert outcome.status == "failed"
    assert "cannot load this config" in outcome.detail
    assert world.current() == "v3.1.0"
    assert world.restarts == 0
    # Nothing half-made is left to be switched to later.
    assert not world.layout.release("v3.2.0").exists()
    assert any("staying on v3.1.0" in line for line in world.said)


def test_a_failed_install_leaves_the_current_release_running(world: World) -> None:
    world.release("3.2.0")

    def no_uv(release: Path, extras: Sequence[str]) -> None:
        raise UpdateError("uv sync failed (1): resolution failed")

    outcome = world.apply("3.1.0", prepare=no_uv)

    assert outcome.status == "failed"
    assert world.current() == "v3.1.0"
    assert not world.layout.release("v3.2.0").exists()


def test_a_restart_that_fails_switches_back(world: World) -> None:
    world.release("3.2.0")

    def broken_restart() -> None:
        raise UpdateError("systemctl try-restart failed")

    outcome = world.apply("3.1.0", restart=broken_restart)

    assert outcome.status == "failed"
    assert world.current() == "v3.1.0"


def test_a_tag_moved_after_it_was_installed_is_not_trusted(world: World) -> None:
    world.release("3.2.0")
    ready = world.apply("3.1.0", restart=world.restart)
    assert ready.status == "applied"
    updater.rollback(world.layout, state_directory=world.state, restart=world.restart)
    # The tag is moved to another commit upstream.
    (world.work / "VERSION").write_text("3.2.0-moved\n")
    git("commit", "--quiet", "-am", "moved", cwd=world.work)
    git("tag", "--force", "v3.2.0", cwd=world.work)
    git("push", "--quiet", "--force", "origin", "main", "--tags", cwd=world.work)

    outcome = world.apply("3.1.0")

    assert outcome.status == "failed"
    assert "moved" in outcome.detail
    assert world.current() == "v3.1.0"


# --- never mid-cycle --------------------------------------------------------------


def test_an_update_waits_for_a_running_cycle_then_switches(world: World) -> None:
    world.release("3.2.0")
    order: list[str] = []
    held = threading.Event()
    finish = threading.Event()

    def cycle() -> None:
        with cycle_lock(world.state) as taken:
            assert taken
            held.set()
            finish.wait(timeout=10)
            order.append("cycle finished")

    worker = threading.Thread(target=cycle)
    worker.start()
    assert held.wait(timeout=10)

    def sleep(_seconds: float) -> None:
        # Called while the update is waiting on the lock: let the cycle end.
        finish.set()
        worker.join(timeout=10)

    def restart() -> None:
        order.append("switched")

    outcome = world.apply("3.1.0", restart=restart, sleep=sleep)

    assert outcome.status == "applied"
    assert order == ["cycle finished", "switched"]
    assert any("waiting for it to finish" in line for line in world.said)


def test_a_scheduled_check_does_not_wait_and_switches_next_time(world: World) -> None:
    world.release("3.2.0")
    with cycle_lock(world.state) as taken:
        assert taken
        busy = world.apply("3.1.0", wait=False)

    assert busy.status == "busy"
    assert world.current() == "v3.1.0"
    # Installed and checked once; the next check only switches.
    assert world.layout.release("v3.2.0").is_dir()
    checks = len(world.checked)
    assert world.apply("3.1.0", wait=False).status == "applied"
    assert len(world.checked) == checks
    assert world.current() == "v3.2.0"


def test_the_cycle_lock_is_exclusive_even_inside_one_process(tmp_path: Path) -> None:
    """The schedule loop runs its cycle on a thread beside the update check."""
    with cycle_lock(tmp_path) as first:
        with cycle_lock(tmp_path, wait=False) as second:
            assert (first, second) == (True, False)


# --- going back --------------------------------------------------------------------


def test_rollback_returns_to_the_previous_release(world: World) -> None:
    world.release("3.2.0")
    world.apply("3.1.0")

    back = updater.rollback(world.layout, state_directory=world.state, restart=world.restart)

    assert back == "v3.1.0"
    assert (world.current(), world.previous()) == ("v3.1.0", "v3.2.0")
    assert world.restarts == 2


def test_rollback_reaches_a_release_installed_by_hand(world: World) -> None:
    """The release an operator started on was set up by hand, without a marker."""
    (world.layout.release("v3.1.0") / updater.READY_MARKER).unlink()
    world.release("3.2.0")
    world.apply("3.1.0")

    assert updater.rollback(world.layout, state_directory=world.state, restart=world.restart) == (
        "v3.1.0"
    )


def test_rollback_with_nothing_to_return_to_is_refused(world: World) -> None:
    with pytest.raises(UpdateError, match="no previous release"):
        updater.rollback(world.layout, state_directory=world.state, restart=world.restart)
    assert world.current() == "v3.1.0"


def test_rollback_waits_for_a_running_cycle_too(world: World) -> None:
    world.release("3.2.0")
    world.apply("3.1.0")
    with cycle_lock(world.state), pytest.raises(UpdateError, match="cycle is running"):
        updater.rollback(
            world.layout, state_directory=world.state, restart=world.restart, wait=False
        )
    assert world.current() == "v3.2.0"


# --- the layout ---------------------------------------------------------------------


def test_the_switch_replaces_the_link_and_leaves_nothing_behind(world: World) -> None:
    world.release("3.2.0")
    world.apply("3.1.0")

    current = world.layout.current
    assert current.is_symlink()
    assert os.readlink(current) == str(Path("releases") / "v3.2.0")
    assert not list(world.layout.root.glob(".*.tmp"))


def test_an_install_is_found_from_where_its_code_runs(tmp_path: Path) -> None:
    package = tmp_path / "prometheon" / "releases" / "v3.1.0" / "src" / "prometheon"
    package.mkdir(parents=True)
    layout = updater.installed_layout(str(package / "__init__.py"))

    assert layout is not None
    assert layout.root == tmp_path / "prometheon"


def test_a_plain_clone_is_not_an_install_that_updates(tmp_path: Path) -> None:
    package = tmp_path / "prometheon_v2" / "src" / "prometheon"
    package.mkdir(parents=True)

    assert updater.installed_layout(str(package / "__init__.py")) is None


def test_versions_compare_as_numbers() -> None:
    assert updater.parse_version("3.10.0") > updater.parse_version("3.9.9")
    assert updater.release_version("v3.10.0") == (3, 10, 0)
    with pytest.raises(UpdateError):
        updater.parse_version("3.1")
