"""``prometheon update``: status, check, apply and rollback.

The work is :mod:`prometheon.update`. This module finds the install the running
code belongs to, reads the config, and decides what a restart means here.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from collections.abc import Callable
from pathlib import Path

import prometheon
from prometheon import update as updater
from prometheon.cli._common import EXIT_FAILED, EXIT_OK, load, note, out, table
from prometheon.config import Config, ValidatorMode
from prometheon.errors import ConfigError, UpdateError
from prometheon.version import __version__


def _layout() -> updater.Layout:
    layout = updater.installed_layout(prometheon.__file__)
    if layout is None:
        raise UpdateError(
            "this copy of prometheon is not installed side by side, so it cannot update "
            "itself. Lay it out as docs/validator.md describes, and run it from <root>/current"
        )
    return layout


def _state_directory(config: Config) -> Path:
    """The state directory, which must outlive any one release."""
    path = config.validator.state_directory
    if not path.is_absolute():
        raise ConfigError(
            f"[validator] state_directory is {str(path)!r}, a relative path, so each release "
            "would keep its own saved vector. Set an absolute path, such as "
            "/var/lib/prometheon, before updating"
        )
    return path


def _extras(config: Config) -> tuple[str, ...]:
    """Score mode runs models; copy mode installs nothing to run them with."""
    return ("wrapper",) if config.validator.mode is ValidatorMode.SCORE else ()


def self_check_release(release: Path, *, config_path: Path) -> None:
    """Run the new release's own self-check, from its own environment."""
    updater.run_command(
        [
            str(release / ".venv" / "bin" / "prometheon"),
            "update",
            "self-check",
            "--config",
            str(config_path),
        ],
        release,
    )


def cmd_status(args: argparse.Namespace) -> int:
    config = load(args.config)
    layout = updater.installed_layout(prometheon.__file__)
    rows = [
        ("running", f"v{__version__}"),
        ("automatic updates", "on" if config.update.auto else "off"),
    ]
    if layout is None:
        rows.append(("install", "not side by side; updates are manual"))
    else:
        known = updater.release_tags(layout)
        newest = updater.newest_release(known, updater.parse_version(__version__))
        rows += [
            ("root", str(layout.root)),
            ("current", layout.target(layout.current) or "-"),
            ("previous", layout.target(layout.previous) or "-"),
            ("newer, already fetched", newest[1] if newest else "none"),
        ]
    out(table(rows))
    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    load(args.config)
    layout = _layout()
    updater.fetch_tags(layout)
    running = updater.parse_version(__version__)
    newest = updater.newest_release(updater.release_tags(layout), running)
    out(
        table(
            [
                ("running", f"v{__version__}"),
                ("newest release", newest[1] if newest else "up to date"),
            ]
        )
    )
    return EXIT_OK


def cmd_apply(args: argparse.Namespace) -> int:
    config = load(args.config)
    if args.scheduled and not config.update.auto:
        note("[update] auto is off: nothing to do")
        return EXIT_OK
    layout = _layout()
    outcome = updater.apply(
        layout,
        running=updater.parse_version(__version__),
        state_directory=_state_directory(config),
        extras=_extras(config),
        self_check=functools.partial(self_check_release, config_path=Path(args.config).resolve()),
        restart=functools.partial(updater.restart_units, config.update.restart_units),
        # The timer runs again in thirty minutes; a manual apply waits.
        wait=not args.scheduled,
        say=note,
    )
    if outcome.status == "current":
        note(f"v{__version__} is the newest release")
    return EXIT_FAILED if outcome.status == "failed" else EXIT_OK


def cmd_rollback(args: argparse.Namespace) -> int:
    config = load(args.config)
    layout = _layout()
    updater.rollback(
        layout,
        state_directory=_state_directory(config),
        restart=functools.partial(updater.restart_units, config.update.restart_units),
        say=note,
    )
    return EXIT_OK


def cmd_self_check(args: argparse.Namespace) -> int:
    """Run inside a new release before anything switches to it."""
    config = load(args.config)
    if not config.hub.epoch_keys:
        raise ConfigError("no hub key is pinned under [hub.epoch_keys]")
    if config.validator.mode is ValidatorMode.SCORE:
        from prometheon.evaluation.preflight import check_evaluation_runtime

        check_evaluation_runtime(device="cpu", dtype="float32")
    note(f"v{__version__}: imports, config, hub key and runtime are fine")
    return EXIT_OK


def scheduled_update(
    config: Config,
    *,
    config_path: Path,
    execv: Callable[[str, list[str]], object] = os.execv,
    prepare: updater.Prepare = updater.uv_sync,
    self_check: updater.SelfCheck | None = None,
) -> None:
    """The 30-minute check inside ``validator schedule``, when ``[update] auto`` is on.

    Never waits for a cycle: the loop only calls it while none runs here, and
    one running elsewhere holds the lock, so this skips to the next boundary.
    After a switch, the loop re-executes itself from the new release, with the
    arguments it was started with.
    """
    layout = updater.installed_layout(prometheon.__file__)
    if layout is None:
        note(
            "automatic updates are on, but this install is not side by side; see docs/validator.md"
        )
        return
    outcome = updater.apply(
        layout,
        running=updater.parse_version(__version__),
        state_directory=_state_directory(config),
        extras=_extras(config),
        self_check=self_check or functools.partial(self_check_release, config_path=config_path),
        restart=lambda: None,
        prepare=prepare,
        wait=False,
        say=note,
    )
    if outcome.status != "applied":
        return
    executable = layout.current / ".venv" / "bin" / "prometheon"
    note(f"re-executing from {outcome.tag}")
    os.chdir(layout.current)
    execv(str(executable), [str(executable), *sys.argv[1:]])
