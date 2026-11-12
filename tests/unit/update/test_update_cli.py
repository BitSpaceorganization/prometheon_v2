"""`prometheon update` and the schedule loop's check, as an operator runs them."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prometheon.cli import update as cli
from prometheon.config import ValidatorMode

from .conftest import World

pytestmark = pytest.mark.unit


def _config(world: World, *, auto: bool = True, mode: ValidatorMode = ValidatorMode.SCORE) -> Any:
    return SimpleNamespace(
        update=SimpleNamespace(auto=auto, restart_units=()),
        validator=SimpleNamespace(state_directory=world.state, mode=mode),
    )


def test_the_timer_does_nothing_while_automatic_updates_are_off(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in: the shipped timer runs every 30 minutes, and does nothing unless asked."""
    monkeypatch.setattr(cli, "load", lambda _path: _config(world, auto=False))

    def never(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an update was attempted with [update] auto off")

    monkeypatch.setattr(cli.updater, "installed_layout", never)
    monkeypatch.setattr(cli.updater, "apply", never)

    code = cli.cmd_apply(argparse.Namespace(config="cfg.toml", scheduled=True))

    assert code == 0


def test_the_schedule_loop_re_executes_itself_from_the_new_release(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.release("3.2.0")
    monkeypatch.setattr(cli, "__version__", "3.1.0")
    monkeypatch.setattr(cli.updater, "installed_layout", lambda _file: world.layout)
    monkeypatch.setattr(cli.sys, "argv", ["prometheon", "validator", "schedule", "--config", "c"])
    monkeypatch.chdir(world.state.parent)
    executed: list[tuple[str, list[str]]] = []

    cli.scheduled_update(
        _config(world),
        config_path=Path("c"),
        execv=lambda path, argv: executed.append((path, argv)),
        prepare=world.prepare,
        self_check=world.self_check,
    )

    assert world.current() == "v3.2.0"
    executable = str(world.layout.current / ".venv" / "bin" / "prometheon")
    assert executed == [(executable, [executable, "validator", "schedule", "--config", "c"])]
    # Score mode installs the runtime that runs models; copy mode would not.
    assert ("v3.2.0", ("wrapper",)) in world.prepared


def test_the_schedule_loop_stays_put_when_nothing_is_newer(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "__version__", "3.1.0")
    monkeypatch.setattr(cli.updater, "installed_layout", lambda _file: world.layout)
    executed: list[Any] = []

    cli.scheduled_update(
        _config(world),
        config_path=Path("c"),
        execv=lambda *args: executed.append(args),
        prepare=world.prepare,
        self_check=world.self_check,
    )

    assert executed == []
    assert world.current() == "v3.1.0"


def test_copy_mode_installs_no_model_runtime(world: World) -> None:
    assert cli._extras(_config(world, mode=ValidatorMode.COPY)) == ()
    assert cli._extras(_config(world)) == ("wrapper",)


def test_a_relative_state_directory_is_refused_before_anything_moves(world: World) -> None:
    """Each release has its own directory, so a relative path would be left behind."""
    config = _config(world)
    config.validator.state_directory = Path(".validator-state")

    with pytest.raises(cli.ConfigError, match="absolute path"):
        cli._state_directory(config)
