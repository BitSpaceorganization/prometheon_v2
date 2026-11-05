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
from prometheon.errors import ConfigError, OutdatedConfigError
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


def _configs() -> list[Path]:
    return sorted((ROOT / "configs").glob("*.example.toml"))


def _filled_in(path: Path, tmp_path: Path) -> Path:
    """The example as an operator would have it, with a netuid filled in."""
    copy = tmp_path / path.name
    copy.write_text(
        path.read_text(encoding="utf-8").replace("\nnetuid = 0\n", "\nnetuid = 7\n"),
        encoding="utf-8",
    )
    return copy


class TestTheShippedConfigsLoad:
    """`extra="forbid"` makes a stale example config a hard failure.

    A new operator's first action is to copy one of these. If it names a field
    the schema dropped, or omits one it now requires, that copy does not load
    and the first experience of the subnet is a validation error.
    """

    def test_at_least_one_example_is_shipped(self) -> None:
        assert _configs(), "configs/*.example.toml is how a new operator starts"

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_example_refuses_to_load_until_a_netuid_is_filled_in(self, path: Path) -> None:
        """No netuid is announced yet, so a copy must not point anywhere by accident."""
        with pytest.raises(ConfigError, match="netuid"):
            load_config(path)

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_example_loads_once_filled_in(self, path: Path, tmp_path: Path) -> None:
        assert isinstance(load_config(_filled_in(path, tmp_path)), Config)

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_example_names_the_rules_it_was_written_for(self, path: Path) -> None:
        assert f'scoring_version = "{SCORING_VERSION}"' in path.read_text(encoding="utf-8")

    def test_no_example_names_a_netuid_as_prometheons(self) -> None:
        """Mainnet 108 belongs to another project; testnet is not used."""
        import re

        for path in _configs():
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"\b(108|481)\b", text), path.name

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
    def test_the_labelling_temperature_suits_the_model_it_ships_with(
        self, path: Path, tmp_path: Path
    ) -> None:
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
        labelling = load_config(_filled_in(path, tmp_path)).labelling
        if not labelling.model.startswith("gpt-5"):
            return
        assert labelling.temperature != 0, (
            f"{path.name} pairs model={labelling.model!r} with temperature=0, "
            "which that model rejects with HTTP 400"
        )

    @pytest.mark.parametrize("path", _configs(), ids=lambda p: p.name)
    def test_the_shipped_split_is_the_agreed_one(self, path: Path, tmp_path: Path) -> None:
        """70% data to the top 10, then 25%, 4% and 1% of total to three models.

        These numbers are the subnet's economics, and they are *config*, not
        code: two validators running identical code but different values
        compute different weight vectors and disagree on chain. Pinning them
        here means a change has to be deliberate and has to land in every
        shipped example at once. Together they cover 100%, so nothing burns
        unless a share goes unclaimed.
        """
        scoring = load_config(_filled_in(path, tmp_path)).scoring
        assert scoring.dataset_share_bp == 7000, path.name
        assert scoring.dataset_top_n == 10, path.name
        assert scoring.model_rank_shares_bp == (2500, 400, 100), path.name
        assert scoring.dataset_share_bp + sum(scoring.model_rank_shares_bp) == 10_000


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


class TestTheDocumentedCommandsExist:
    """Docs that name a command the parser does not have send users nowhere.

    This has drifted more than once. The README documented `account link`,
    which is a platform-side step with no subnet command, and `model deploy`
    and `model render`, neither of which survived the move to validator-side
    evaluation.
    """

    def _documented(self) -> set[tuple[str, ...]]:
        """Every ``prometheon <group> <sub>`` named in a shipped file.

        The example configs are scanned alongside the prose. They name commands
        too, and because nothing checked them they went on advising
        ``prometheon model render`` for months after that command was removed.
        """
        import re

        found: set[tuple[str, ...]] = set()
        for path in [
            ROOT / "README.md",
            *sorted((ROOT / "docs").glob("*.md")),
            *sorted((ROOT / "configs").glob("*.toml")),
        ]:
            for match in re.finditer(
                r"prometheon\s+([a-z-]+)(?:\s+([a-z-]+))?", path.read_text(encoding="utf-8")
            ):
                group, sub = match.group(1), match.group(2)
                if sub and sub not in {"--config", "--help"}:
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


class TestAnOldConfigFailsLoudly:
    """Operators copy an example once and keep it, so an old copy must not run.

    A changed default in code never reaches a copied file. Without this, a v2
    file would run v3 code on v2 numbers, or fail on whichever renamed field
    the parser met first.
    """

    def _write(self, tmp_path: Path, text: str) -> Path:
        path = tmp_path / "old.toml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_v2_config_is_refused_with_what_to_do(self, tmp_path: Path) -> None:
        v2 = (
            '[chain]\nnetwork = "finney"\nnetuid = 108\n'
            '[wallet]\nname = "v"\nhotkey = "h"\n'
            '[db]\nbase_url = "https://audit.bitfan.ai"\n'
            "[scoring]\nminer_burn_share_bp = 0\ndataset_share_bp = 5000\n"
        )
        with pytest.raises(OutdatedConfigError, match="v2 rules"):
            load_config(self._write(tmp_path, v2))

    def test_a_config_for_other_rules_is_refused(self, tmp_path: Path) -> None:
        text = (ROOT / "configs" / "finney.example.toml").read_text(encoding="utf-8")
        older = text.replace(SCORING_VERSION, "prometheon-scoring/2.1").replace(
            "\nnetuid = 0\n", "\nnetuid = 7\n"
        )
        with pytest.raises(OutdatedConfigError, match=r"prometheon-scoring/2\.1"):
            load_config(self._write(tmp_path, older))


class TestTheDocsNameNoNetuid:
    """The subnet has no netuid yet, and mainnet 108 belongs to another project.

    Operators copy what the guides show, so a netuid in prose becomes a netuid
    in somebody's config. None may appear until the team announces one, and
    then it belongs in config rather than in the guides. The configs have their
    own check above.
    """

    @staticmethod
    def _documents() -> list[Path]:
        candidates = [
            *ROOT.glob("*.md"),
            *(ROOT / "docs").glob("*.md"),
            *(ROOT / "deploy").rglob("*"),
        ]
        return sorted(path for path in candidates if path.is_file())

    def test_the_guides_are_scanned_at_all(self) -> None:
        names = {path.name for path in self._documents()}
        assert {"README.md", "validator.md", "miner.md", "scoring.md"} <= names

    def test_no_guide_names_108_or_481(self) -> None:
        import re

        offenders = [
            f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
            for path in self._documents()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if re.search(r"\b(108|481)\b", line)
        ]
        assert not offenders, "\n".join(offenders)

    def test_no_guide_sends_operators_to_testnet(self) -> None:
        """The team doesn't use testnet; the local chain path replaces it."""
        for path in self._documents():
            text = path.read_text(encoding="utf-8").lower()
            for advice in ("testnet first", "--network test", 'network = "test"'):
                assert advice not in text, f"{path.relative_to(ROOT)} says {advice!r}"
