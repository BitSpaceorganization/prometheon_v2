"""A cycle holds the cycle lock for its whole run, so an update never splits one."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prometheon.cli import validator as cli
from prometheon.config import ValidatorMode
from prometheon.lock import cycle_lock

pytestmark = pytest.mark.unit


def _config(state: Path, mode: ValidatorMode) -> Any:
    return SimpleNamespace(validator=SimpleNamespace(state_directory=state, mode=mode))


@pytest.mark.parametrize("mode", [ValidatorMode.SCORE, ValidatorMode.COPY])
def test_a_running_cycle_holds_the_lock_an_update_takes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: ValidatorMode
) -> None:
    seen: list[bool] = []

    def body(*_args: Any, **_kwargs: Any) -> int:
        # What an update would find while the cycle runs.
        with cycle_lock(tmp_path, wait=False) as taken:
            seen.append(taken)
        return 0

    monkeypatch.setattr(cli, "load", lambda _path: _config(tmp_path, mode))
    monkeypatch.setattr(cli, "_score_the_day", body)
    monkeypatch.setattr(cli, "cmd_copy", body)

    assert cli.cmd_run(argparse.Namespace(config="cfg.toml", date=None, force=False)) == 0
    assert seen == [False]
    # And it is free again once the cycle returns.
    with cycle_lock(tmp_path, wait=False) as taken:
        assert taken
