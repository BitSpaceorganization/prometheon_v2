"""Validator CLI plumbing that is easy to get wrong and expensive to get wrong."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

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
