"""The files this repository ships must actually work.

Every check here failed at least once during development, and each failure was
the same shape: an artefact a user touches on their first run drifting away from
the code that reads it. None of them is caught by a unit test, because each side
is individually correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheon.canonical.hashes import (
    ACCEPTED_WRAPPER_HASHES,
    IMAGE_NAME,
    IMAGE_TAG,
    IMAGE_USERNAME,
    accepted_image_ids,
    chute_id_for,
    image_id_for,
)
from prometheon.canonical.integrity import (
    canonical_wrapper_hash,
    render_wrapper,
    wrapper_hash,
)
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
#: wrapper hash was derived from `ast.dump`, whose output changed in CPython
#: 3.12 and again in 3.13, so the same source hashed to three different values
#: across the supported range while every test stayed green.
#:
#: These two lines are the only place the expected value exists independently
#: of the code that produces it. Changing either one changes what every
#: validator on the network accepts, so a diff that touches them is a
#: subnet-wide migration and must follow docs/canonical-wrapper.md.
EXPECTED_SCRIPT_HASH = "ea4dab57ccd0c228337318d141c4eb1eff973ce44b36f02d1bba8a9792d08af7"


class TestTheCanonicalArtifactsAgree:
    """If these drift, every deployed model on the subnet becomes invalid at once."""

    def test_the_wrapper_hash_is_the_published_value(self) -> None:
        """Pinned against a literal, and run on every Python in the CI matrix.

        This is the test that would have caught the `ast.dump` instability.
        """
        assert canonical_wrapper_hash() == EXPECTED_SCRIPT_HASH

    def test_normalisation_does_not_depend_on_the_interpreter(self) -> None:
        """The property the literals above defend, stated directly.

        `normalise_source` must emit only node types and fields this package
        names itself. A field that exists on one CPython release and not
        another (`type_params`, added 3.12) must never reach the output.
        """
        from prometheon.canonical.integrity import normalise_source

        rendered = render_wrapper(
            hf_repo="a/b", hf_revision="0" * 40, chutes_user="u", hotkey="5H", image_id="i"
        )
        normalised = normalise_source(rendered)
        for moved in ("type_params", "type_comment", "type_ignores", "kind="):
            assert moved not in normalised, f"{moved} is version-dependent and must not be emitted"

    def test_an_unknown_construct_is_refused_rather_than_hashed(self) -> None:
        """The allowlist doubles as a check on what a deployment may contain."""
        from prometheon.canonical.integrity import normalise_source
        from prometheon.errors import RegistryError

        with pytest.raises(RegistryError, match="canonical wrapper grammar"):
            normalise_source("match x:\n    case 1:\n        pass\n")

    def test_a_rendered_wrapper_hashes_to_the_canonical_value(self) -> None:
        rendered = render_wrapper(
            hf_repo="example/model",
            hf_revision="0" * 40,
            chutes_user="example",
            hotkey="5Hot",
            image_id="img-1",
        )
        assert wrapper_hash(rendered) == canonical_wrapper_hash()

    def test_the_miner_supplied_values_do_not_change_the_hash(self) -> None:
        """Normalisation is what lets four values vary without breaking the pin."""
        first = wrapper_hash(
            render_wrapper(
                hf_repo="a/one",
                hf_revision="a" * 40,
                chutes_user="alice",
                hotkey="5Alice",
                image_id="img-1",
                gpu_count=1,
                min_vram_gb=16,
            )
        )
        second = wrapper_hash(
            render_wrapper(
                hf_repo="b/two",
                hf_revision="b" * 40,
                chutes_user="bob",
                hotkey="5Bob",
                image_id="img-1",
                # A different GPU choice must not change the hash either: it is
                # the miner's to make and they pay for it.
                gpu_count=8,
                min_vram_gb=140,
            )
        )
        assert first == second == canonical_wrapper_hash()

    def test_the_canonical_hash_is_one_validators_accept(self) -> None:
        assert canonical_wrapper_hash() in ACCEPTED_WRAPPER_HASHES

    def test_the_subnet_image_is_configured(self) -> None:
        """An unconfigured image means no deployment can be verified at all.

        `accepted_image_ids()` is empty until the account strings are set, and
        the registry treats an empty set as "refuse everything" — the safe
        direction, since the Chutes API exposes no image contents and an
        unverified image is an unknown runtime. This fails loudly rather than
        letting a build ship that would zero the whole field.
        """
        assert IMAGE_USERNAME and IMAGE_NAME and IMAGE_TAG
        assert accepted_image_ids() == frozenset(
            {image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG)}
        )

    def test_the_chute_id_is_derivable_before_the_chute_exists(self) -> None:
        """Which is why the script names the chute from the hotkey.

        A miner must commit `chute_id` before deploying, and a miner-chosen name
        made that impossible: the id did not exist yet. Derived, it is knowable
        in advance and checkable afterwards.
        """
        assert chute_id_for("alice", "5Hot") == chute_id_for("alice", "5Hot")
        assert chute_id_for("alice", "5Hot") != chute_id_for("bob", "5Hot")
        assert chute_id_for("alice", "5Hot") != chute_id_for("alice", "5Cold")


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
