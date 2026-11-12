"""A real git origin with release tags, and an install laid out side by side."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from prometheon import update as updater

GIT = shutil.which("git") or "git"


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(  # noqa: S603 - a fixed git argv in a test
        [GIT, *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture(autouse=True)
def _plain_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No global git config: no signing, no hooks, the same on every machine."""
    empty = tmp_path / "gitconfig"
    empty.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")


class World:
    """An origin with releases, an install of v3.1.0, and recorders for the rest."""

    def __init__(self, tmp_path: Path) -> None:
        self.origin = tmp_path / "origin.git"
        self.work = tmp_path / "work"
        git("init", "--quiet", "--bare", str(self.origin), cwd=tmp_path)
        git("init", "--quiet", "-b", "main", str(self.work), cwd=tmp_path)
        git("remote", "add", "origin", str(self.origin), cwd=self.work)
        self.layout = updater.Layout(root=tmp_path / "prometheon")
        self.state = tmp_path / "state"
        self.prepared: list[tuple[str, tuple[str, ...]]] = []
        self.checked: list[str] = []
        self.restarts = 0
        self.said: list[str] = []
        self.release("3.1.0")
        git("clone", "--quiet", str(self.origin), str(self.layout.repo), cwd=tmp_path)
        updater.install(
            self.layout, "v3.1.0", extras=(), prepare=self.prepare, self_check=self.self_check
        )
        updater.switch(self.layout, "v3.1.0")

    def release(self, version: str, *, tag: str | None = None) -> None:
        """Commit a version and push it with a tag, as the team publishes one."""
        (self.work / "VERSION").write_text(version + "\n")
        git("add", "VERSION", cwd=self.work)
        git("commit", "--quiet", "-m", f"release {version}", cwd=self.work)
        git("tag", tag or f"v{version}", cwd=self.work)
        git("push", "--quiet", "origin", "main", "--tags", cwd=self.work)

    def prepare(self, release: Path, extras: Sequence[str]) -> None:
        self.prepared.append((release.name, tuple(extras)))

    def self_check(self, release: Path) -> None:
        self.checked.append(release.name)

    def restart(self) -> None:
        self.restarts += 1

    def apply(self, running: str, **overrides: Any) -> updater.Outcome:
        options: dict[str, Any] = {
            "running": updater.parse_version(running),
            "state_directory": self.state,
            "extras": ("wrapper",),
            "prepare": self.prepare,
            "self_check": self.self_check,
            "restart": self.restart,
            "say": self.said.append,
        }
        options.update(overrides)
        return updater.apply(self.layout, **options)

    def current(self) -> str | None:
        return self.layout.target(self.layout.current)

    def previous(self) -> str | None:
        return self.layout.target(self.layout.previous)


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)
