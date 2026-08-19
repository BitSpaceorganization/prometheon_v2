"""The files this repository ships must actually work.

Every check here failed at least once during development, and each failure was
the same shape: an artefact a user touches on their first run drifting away from
the code that reads it. None of them is caught by a unit test, because each side
is individually correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheon.cli.main import build_parser
from prometheon.cli.validator import _policy_version
from prometheon.config import Config, ScoringConfig, load_config

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


def _configs() -> list[Path]:
    return sorted((ROOT / "configs").glob("*.example.toml"))


class TestTheShippedConfigsLoad:
    """`extra="forbid"` makes a stale example config a hard failure.

    A new operator's first action is to copy one of these. If it names a field
    the schema dropped, or omits one it now requires, that copy does not load
    and the first experience of the subnet is a validation error.
    """

    def test_at_least_one_example_is_shipped(self) -> None:
        assert _configs(), "configs/*.example.toml is how a new operator starts"

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_example_loads(self, path: Path) -> None:
        assert isinstance(load_config(path), Config)

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_every_emission_affecting_field_is_visible(self, path: Path) -> None:
        """`[scoring]` decides payouts, so none of it may be invisible.

        A field that exists only as a default is a number nobody reviewing a
        config can see, and this is the section where an unseen number is a
        change in what miners get paid.
        """
        text = path.read_text(encoding="utf-8")
        missing = [name for name in ScoringConfig.model_fields if name not in text]
        assert not missing, f"{path.name} does not mention {missing}"

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_labelling_temperature_suits_the_model_it_ships_with(self, path: Path) -> None:
        """A shipped config must survive its own first cycle.

        The blocker: the examples named a gpt-5 reasoning model and left
        `temperature` unset, so it defaulted to 0 -- which that line rejects
        outright::

            Unsupported value: 'temperature' does not support 0 with this
            model. Only the default (1) value is supported.

        Every validator copying the example therefore died at labelling, hours
        into a cycle and after paying for the snapshot. `temperature = 0` is
        still the right default for a model that accepts it, because it is what
        stops ground truth wandering between validators; it just cannot be the
        shipped one here.
        """
        labelling = load_config(path).labelling
        if not labelling.model.startswith("gpt-5"):
            return
        assert labelling.temperature != 0, (
            f"{path.name} pairs model={labelling.model!r} with temperature=0, "
            "which that model rejects with HTTP 400"
        )

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_shipped_split_is_the_agreed_one(self, path: Path) -> None:
        """40% burns, 40% pays for data, 20% pays for models.

        These three numbers are the subnet's economics, and they are *config*,
        not code: two validators running identical code but different values
        compute different weight vectors and disagree on chain. Pinning them
        here means a change has to be deliberate and has to land in every
        shipped example at once.
        """
        scoring = load_config(path).scoring
        total = 1_000_000
        pool = total * (10_000 - scoring.miner_burn_share_bp) // 10_000
        dataset = pool * scoring.dataset_share_bp // 10_000
        model = pool - dataset
        burn = total - pool

        assert burn / total == pytest.approx(0.40, abs=0.001), f"{path.name}: burn"
        assert dataset / total == pytest.approx(0.40, abs=0.001), f"{path.name}: dataset"
        assert model / total == pytest.approx(0.20, abs=0.001), f"{path.name}: model"


class TestTheShippedPolicyIsUsable:
    def test_it_exists(self) -> None:
        assert (ROOT / "content_policy.md").is_file()

    def test_it_declares_a_version_the_runtime_can_read(self) -> None:
        """The blocker this test exists for.

        `validator run` refuses to start without a policy version, and the
        policy writes it as Markdown (`**Version:** …`). A parser that only
        matched a bare `Version:` made the shipped repository unrunnable.
        """
        assert _policy_version((ROOT / "content_policy.md").read_text(encoding="utf-8"))


#: The published pins. Literals, not recomputed values.
#:
#: Every other assertion in this file compares one computed hash against
#: another computed hash, which proves the code agrees with itself and nothing
#: more. That is exactly how a subnet-breaking defect passed 796 tests: the
EXPECTED_SCRIPT_HASH = "ea4dab57ccd0c228337318d141c4eb1eff973ce44b36f02d1bba8a9792d08af7"


class TestTheDocumentedCommandsExist:
    """Docs that name a command the parser does not have send users nowhere.

    Two had already drifted: the README documented `account link`, which is a
    platform-side step with no subnet command, and `model deploy`, which is
    called `model render`.
    """

    def _documented(self) -> set[tuple[str, ...]]:
        import re

        found: set[tuple[str, ...]] = set()
        for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
            for match in re.finditer(
                r"prometheon\s+([a-z-]+)(?:\s+([a-z-]+))?", path.read_text(encoding="utf-8")
            ):
                group, sub = match.group(1), match.group(2)
                if group in {"canonical"}:
                    found.add((group,))
                elif sub and sub not in {"--config", "--help"}:
                    found.add((group, sub))
        return found

    def test_every_documented_command_is_a_real_one(self) -> None:
        parser = build_parser()
        groups = parser._subparsers._group_actions[0].choices

        for command in sorted(self._documented()):
            assert command[0] in groups, (
                f"docs name `prometheon {command[0]}`, which does not exist"
            )
            if len(command) == 2:
                sub = getattr(groups[command[0]], "_subparsers", None)
                assert sub is not None, f"`prometheon {command[0]}` takes no subcommand"
                assert command[1] in sub._group_actions[0].choices, (
                    f"docs name `prometheon {command[0]} {command[1]}`, which does not exist"
                )


class TestTheDocumentationLinksResolve:
    def test_every_relative_markdown_link_points_at_a_file(self) -> None:
        import re

        broken: list[str] = []
        for path in [
            ROOT / "README.md",
            ROOT / "CONTRIBUTING.md",
            ROOT / "SECURITY.md",
            *sorted((ROOT / "docs").glob("*.md")),
        ]:
            for match in re.finditer(
                r"\]\((\.{0,2}/[^)#]+|[A-Za-z0-9_][^):#]*\.md)\)", path.read_text(encoding="utf-8")
            ):
                target = (path.parent / match.group(1)).resolve()
                if not target.exists():
                    broken.append(f"{path.name} -> {match.group(1)}")
        assert not broken, f"broken links: {broken}"
