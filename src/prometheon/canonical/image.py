"""The subnet's published image, as a definition rather than an id.

Every miner's wrapper references this image by id and builds nothing, which is
what keeps a miner from slipping code into the runtime their weights load
inside. The id is a pure function of ``username/name:tag`` and was knowable long
before the image existed — which is how it came to be referenced by every
wrapper while nothing in this repository could produce it. A miner following
`docs/miner.md` reached `chutes deploy` and was told the image "is not available
to be used (yet)", with nothing to build and no way forward.

This module is the missing half. The constants in `hashes.py` name the image;
this names its contents, so the id every validator computes now corresponds to
something a subnet owner can actually build:

    uv run prometheon image build          # once, from the subnet's account

The dependency set is exactly what the canonical wrapper imports at runtime and
nothing else. An image is not a place to be generous: anything extra here is
code inside the sandbox that no validator reviewed and no hash covers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from prometheon.canonical.hashes import IMAGE_NAME, IMAGE_TAG, IMAGE_USERNAME

if TYPE_CHECKING:  # pragma: no cover - import shape only
    from chutes.image import Image  # type: ignore[import-untyped]

#: Matches the wrapper's own floor. Below this `torch.inference_mode` and the
#: tokenizer APIs the engine uses are not all present.
PYTHON_VERSION: Final[str] = "3.12"

#: Chained via `from_base`, not passed to the constructor. The published docs
#: show `Image(..., python_version=, base_image=)`, but the pinned SDK's
#: constructor takes only username/name/tag/readme — the docs describe a newer
#: release. Follow the installed signature, not the website.
#:
#: Must be a Chutes-known base. Tag ``1`` 500s on the platform for this name;
#: ``parachutes/python:3.12`` is the documented runtime and is what the
#: successful ``1.0`` upload used.
BASE_IMAGE: Final[str] = "parachutes/python:3.12"

#: Pinned, not floated. A rebuild that silently picks up a new torch changes
#: what every model on the subnet runs inside, which is a change to the thing
#: the competition holds constant. Bumping these is an image migration —
#: `LEGACY_IMAGE_IDS` exists to give one an overlap window.
REQUIREMENTS: Final[tuple[str, ...]] = (
    "torch==2.5.1",
    "transformers==4.46.3",
    "huggingface_hub==0.26.2",
    "accelerate==1.1.1",
    "safetensors==0.4.5",
    "fastapi==0.115.5",
    "pydantic==2.10.2",
)


#: Shown on the public image page. It says what the image is *for*, because a
#: miner who finds it needs to know they reference it rather than rebuild it.
README: Final[str] = """\
The Prometheon subnet's published inference runtime.

Every miner's chute references this image by id and builds nothing, so the
runtime a model loads inside is one the subnet published and every validator
can name. Miners supply weights; the engine and its dependencies are fixed.

Do not fork or rebuild this to deploy: a chute referencing any other image is
refused, because a build step on the miner's side is code no hash covers.
"""


def build_image() -> Image:
    """The image every miner's chute references.

    Imported lazily so the SDK stays an optional dependency: a validator scoring
    models never builds anything, and should not need the build toolchain
    installed to run.
    """
    from chutes.image import Image

    # `readme` is required by the platform, not optional as the constructor
    # signature suggests: omitting it fails the upload, not the build.
    image: Any = Image(
        username=IMAGE_USERNAME,
        name=IMAGE_NAME,
        tag=IMAGE_TAG,
        readme=README,
    )
    image = image.from_base(BASE_IMAGE)
    # One command rather than seven: each `run_command` is a layer, and the
    # resolver needs to see the whole set at once to reject an incompatible
    # combination at build time instead of at first inference.
    image = image.run_command("pip install --no-cache-dir " + " ".join(REQUIREMENTS))
    # Inference needs no write access and no privileges. A model that can only
    # read the weights it was given is one fewer thing a crafted input can do
    # something with.
    image = image.set_user("chutes")
    return image


__all__ = ["PYTHON_VERSION", "REQUIREMENTS", "build_image"]
