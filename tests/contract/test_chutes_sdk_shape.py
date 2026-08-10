"""Pins the Chutes SDK shapes the canonical miner script depends on.

**This file exists because its absence produced every critical defect in an
independent review of this repository.** `chutes` was not a dependency, no test
imported it, and the deploy script had therefore never been executed against the
library it targets — only hashed and AST-normalised. The suite proved the script
was *canonical and stable*; it never once proved it *runs*.

Three things followed, each of which this file would have caught:

- ``Image(username, name, tag)`` takes three required positional arguments. The
  script built its image with ``cls()``, so it raised at module import, before a
  line of the moderation path ran. No miner could have deployed at all.
- ``NodeSelector`` is a Pydantic model whose fields are plain integers. The
  script applied config by calling attributes that resolved to callables, so
  every GPU request was silently discarded and every miner got the defaults.
- ``Cord`` takes ``public_api_method``; ``method`` is the internal call. The
  health endpoint published a POST while every caller built a GET.

Everything here is offline: construction and signature introspection, no
network, no account, no deploy. The rule the file encodes is simply that a
boundary with no test that imports the real library is a boundary with bugs —
``test_bittensor_sdk_shape.py`` is the same idea, and the chain adapter is the
one part of this codebase the review found nothing wrong with.
"""

from __future__ import annotations

import inspect
import uuid

import pytest

from prometheon.canonical.hashes import (
    IMAGE_NAME,
    IMAGE_TAG,
    IMAGE_USERNAME,
    accepted_image_ids,
    chute_id_for,
    image_id_for,
)

pytestmark = pytest.mark.contract

chutes = pytest.importorskip("chutes", reason="the Chutes SDK is an optional test dependency")


# ---------------------------------------------------------------------------
# The image is referenced, never built
# ---------------------------------------------------------------------------


def test_a_chute_accepts_an_image_id_string() -> None:
    """The whole design rests on this one annotation.

    A chute may point at an already-built public image by id, so miners build
    nothing and the runtime their weights load inside is one the subnet
    published. If this ever became ``image: Image``, every miner would be back to
    building their own — and the API exposes no image contents, so a validator
    could not tell an honest build from a tampered one.
    """
    from chutes.chute import Chute

    annotation = inspect.signature(Chute.__init__).parameters["image"].annotation
    assert "str" in str(annotation)


def test_building_an_image_needs_three_arguments() -> None:
    """The defect that stopped every deployment, pinned.

    ``Image()`` raises. The script does not construct one at all now, and this
    is what would have said so on the first run rather than on the first deploy.
    """
    from chutes.image import Image

    parameters = inspect.signature(Image.__init__).parameters
    required = [
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.default is inspect.Parameter.empty
    ]
    assert required == ["username", "name", "tag"]

    with pytest.raises(TypeError):
        Image()  # type: ignore[call-arg]


def test_the_image_id_is_derived_from_its_three_strings() -> None:
    """So the id is knowable before the image is ever built.

    Reproduced against a real public image on the live platform:
    ``chutes/sglang:0.5.13-post1c`` → ``a3028618-6f8f-5afc-9a51-8eff25f2e5c8``.
    A mismatch here means ``chutes build`` will publish under an id this repo
    does not expect, and every miner referencing our constant would be refused.
    """
    assert image_id_for("chutes", "sglang", "0.5.13-post1c") == (
        "a3028618-6f8f-5afc-9a51-8eff25f2e5c8"
    )
    assert image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG) in accepted_image_ids()


def test_the_sdk_derives_an_image_id_the_same_way_this_package_does() -> None:
    """Two implementations of one formula, checked against each other."""
    from chutes.image import Image

    built = Image(username="acme", name="widget", tag="7")
    assert built.uid == image_id_for("acme", "widget", "7")


# ---------------------------------------------------------------------------
# The chute's own id, and the name that decides it
# ---------------------------------------------------------------------------


def test_a_chute_id_is_derived_from_its_name() -> None:
    """Which is why the script names the chute after the miner's hotkey.

    A miner-chosen name made the committed ``chute_id`` both unverifiable and
    unknowable: they had to commit an id before any chute existed to have one.
    Derived, the CLI can print it before deploying and a validator can check it
    afterwards.
    """
    assert chute_id_for("alice", "5Hot") == str(
        uuid.uuid5(uuid.NAMESPACE_OID, "alice::chute::prometheon-5Hot")
    )


# ---------------------------------------------------------------------------
# The knobs a miner sets, and the ones the subnet fixes
# ---------------------------------------------------------------------------


def test_node_selector_fields_are_values_and_not_methods() -> None:
    """The defect that silently discarded every GPU request.

    Configuration used to be applied by calling attributes named in a
    miner-supplied YAML, keeping only those that turned out to be callable. A
    Pydantic ``int`` field is not callable, so every key was dropped without a
    word and every miner got one GPU with a 16GB floor whatever they asked for.
    """
    from chutes.chute.node_selector import NodeSelector

    selector = NodeSelector(gpu_count=4, min_vram_gb_per_gpu=80)
    assert (selector.gpu_count, selector.min_vram_gb_per_gpu) == (4, 80)
    assert not callable(selector.gpu_count)


def test_the_gpu_envelope_this_package_offers_matches_the_sdk() -> None:
    """A value the platform would refuse must fail while rendering, not on deploy."""
    from chutes.chute.node_selector import NodeSelector

    from prometheon.canonical.integrity import (
        MAX_GPU_COUNT,
        MAX_VRAM_GB,
        MIN_GPU_COUNT,
        MIN_VRAM_GB,
    )

    fields = NodeSelector.model_fields
    gpu = fields["gpu_count"].metadata
    vram = fields["min_vram_gb_per_gpu"].metadata
    assert (MIN_GPU_COUNT, MAX_GPU_COUNT) == (
        next(m.ge for m in gpu if hasattr(m, "ge")),
        next(m.le for m in gpu if hasattr(m, "le")),
    )
    assert (MIN_VRAM_GB, MAX_VRAM_GB) == (
        next(m.ge for m in vram if hasattr(m, "ge")),
        next(m.le for m in vram if hasattr(m, "le")),
    )


def test_the_sdk_scales_a_chute_to_zero_by_default() -> None:
    """Which is why the script sets these explicitly.

    Eligibility requires a hot chute at cycle time. On the SDK defaults every
    chute in the subnet would be cold between daily cycles, so the steady state
    would be the whole field failing the reachability check — and the logs would
    read as ordinary churn.
    """
    from chutes.chute import Chute

    parameters = inspect.signature(Chute.__init__).parameters
    assert parameters["shutdown_after_seconds"].default == 300
    assert parameters["max_instances"].default == 1
    assert parameters["concurrency"].default == 1


def test_a_cord_publishes_the_verb_named_by_public_api_method() -> None:
    """``method`` is the internal call; the published verb is the other one.

    Setting only ``method`` published a POST route while every caller built a
    GET. Dead code at the time it was written, which is exactly how it survived.
    """
    from chutes.chute.cord import Cord

    parameters = inspect.signature(Cord.__init__).parameters
    assert parameters["public_api_method"].default == "POST"
    assert "method" in parameters


# ---------------------------------------------------------------------------
# The script itself, against the real library
# ---------------------------------------------------------------------------


def test_the_rendered_script_parses_and_declares_what_it_should() -> None:
    """The rendered artifact, read back with the SDK importable.

    Not executed — importing it would download a model — but parsed and read
    for the values a validator checks, with the real library present so an
    import that could not resolve would fail here.
    """
    from prometheon.canonical.integrity import extract_declared_variables, render_wrapper

    source = render_wrapper(
        hf_repo="acme/guard",
        hf_revision="a" * 40,
        chutes_user="acme",
        hotkey="5Hot",
        image_id=image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG),
        gpu_count=2,
        min_vram_gb=80,
    )
    declared = extract_declared_variables(source)
    assert declared["PROMETHEON_HF_REPO"] == "acme/guard"
    assert declared["PROMETHEON_HF_REVISION"] == "a" * 40
    assert declared["PROMETHEON_CHUTES_USER"] == "acme"
    assert declared["PROMETHEON_HOTKEY"] == "5Hot"

    # The image is a plain string in the source, never an `Image(...)` call.
    assert 'image=PROMETHEON_IMAGE_ID' in source
    assert "Image(" not in source
