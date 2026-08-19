"""Validator CLI plumbing that is easy to get wrong and expensive to get wrong."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prometheon.cli.validator import _policy_version, _target_day

pytestmark = pytest.mark.unit


class TestThePolicyVersionIsFoundInARealDocument:
    """The policy is Markdown written for a human, not a config file.

    This is a regression test with a live blocker behind it: the parser
    originally matched `line.startswith("version:")`, the shipped
    `content_policy.md` writes `**Version:** 2026-08-07`, and so
    `prometheon validator run` refused to start on a correct repository.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("**Version:** 2026-08-07", "2026-08-07"),
            ("Version: 1.2.3", "1.2.3"),
            ("*version*: x", "x"),
            ("__Version:__ 2026-01-01", "2026-01-01"),
            ("  Version:   spaced  ", "spaced"),
        ],
    )
    def test_the_forms_a_markdown_policy_actually_uses(self, text: str, expected: str) -> None:
        assert _policy_version(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "no version anywhere",
            "## Version: 9",  # a heading is a section, not a declaration
            "Versioning: what we do",
            "",
        ],
    )
    def test_what_must_not_be_read_as_a_version(self, text: str) -> None:
        assert _policy_version(text) == ""

    def test_the_shipped_policy_declares_one(self) -> None:
        """The check that would have caught the blocker.

        Ties the parser to the actual document rather than to an example of it,
        so editing one and not the other fails here.
        """
        policy = Path(__file__).resolve().parents[3] / "content_policy.md"
        assert policy.is_file(), "content_policy.md is missing from the repository"
        assert _policy_version(policy.read_text(encoding="utf-8"))

    def test_the_first_declaration_wins(self) -> None:
        assert _policy_version("**Version:** 1\n\nlater\n\nVersion: 2") == "1"


class TestTheDefaultDayIsYesterday:
    """Today's content is still arriving; a cycle scores the day that is over."""

    def test_no_date_means_yesterday_utc(self) -> None:
        import argparse

        today = dt.datetime.now(dt.timezone.utc).date()
        assert _target_day(argparse.Namespace(date=None)) == today - dt.timedelta(days=1)

    def test_an_explicit_date_is_honoured(self) -> None:
        import argparse

        assert _target_day(argparse.Namespace(date="2026-08-05")) == dt.date(2026, 8, 5)

    def test_a_malformed_date_is_a_typed_error(self) -> None:
        import argparse

        from prometheon.errors import ConfigError

        with pytest.raises(ConfigError, match="YYYY-MM-DD"):
            _target_day(argparse.Namespace(date="05/08/2026"))


class TestTheEntryPointExitsCorrectly:
    """`main` is the console script, and nothing called it until this test.

    `--version` and `--help` exited 2 because `exc.code or EXIT_USAGE` turns
    argparse's `SystemExit(0)` into 2. They are the first two commands anyone
    runs, and a non-zero exit breaks `set -e` wrappers and healthchecks.
    """

    @pytest.mark.parametrize("argv", [["--version"], ["--help"]])
    def test_informational_flags_exit_zero(
        self, argv: list[str], capsys: pytest.CaptureFixture[str]
    ) -> None:
        from prometheon.cli.main import main

        assert main(argv) == 0
        assert capsys.readouterr().out.strip()

    def test_a_usage_error_still_exits_non_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        from prometheon.cli.main import main

        assert main(["no-such-command"]) != 0

    def test_no_arguments_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        from prometheon.cli.main import main

        assert main([]) != 0


class TestTheAllocationSurvivesBetweenCycles:
    """A cycle runs daily; weights stop counting in hours.

    `activity_cutoff` on netuid 108 is 720 blocks -- under three hours -- while
    a cycle runs once a day. Without the saved vector there is nothing to
    re-post, the validator's row is masked out of consensus for most of every
    day, and the miners it weighted earn nothing.
    """

    def _config(self, tmp_path: Path, netuid: int = 108) -> object:
        """Only the fields the state helpers touch. A real Config here would
        assert nothing extra and would break on every unrelated field added."""
        return SimpleNamespace(
            chain=SimpleNamespace(netuid=netuid),
            validator=SimpleNamespace(state_directory=tmp_path / "state"),
        )

    def _result(self) -> object:
        return SimpleNamespace(
            burn_hotkey="5Hburn",
            split=SimpleNamespace(
                weights={"5Hburn": 800_000, "5Hminer_a": 120_000, "5Hminer_b": 80_000}
            ),
        )

    def test_the_submitted_vector_outlives_the_process_that_computed_it(
        self, tmp_path: Path
    ) -> None:
        from prometheon.cli.validator import _state_path, save_submitted_allocation

        config = self._config(tmp_path)
        path = save_submitted_allocation(config, day=dt.date(2026, 8, 17), result=self._result())

        assert path == _state_path(config)
        state = json.loads(path.read_text(encoding="utf-8"))
        assert state["weights"] == self._result().split.weights
        assert state["burn_hotkey"] == "5Hburn"
        assert state["day"] == "2026-08-17"
        assert state["netuid"] == 108

    def test_the_directory_is_created_rather_than_required(self, tmp_path: Path) -> None:
        """A first cycle on a fresh box must not fail after doing all the work."""
        from prometheon.cli.validator import save_submitted_allocation

        config = self._config(tmp_path / "nested" / "deeper")
        path = save_submitted_allocation(config, day=dt.date(2026, 8, 17), result=self._result())

        assert path.is_file()

    def test_the_netuid_is_recorded_so_a_re_post_cannot_cross_subnets(self, tmp_path: Path) -> None:
        """Re-posting one subnet's payout onto another would pay strangers."""
        from prometheon.cli.validator import save_submitted_allocation

        path = save_submitted_allocation(
            self._config(tmp_path, netuid=108), day=dt.date(2026, 8, 17), result=self._result()
        )

        assert json.loads(path.read_text(encoding="utf-8"))["netuid"] == 108
