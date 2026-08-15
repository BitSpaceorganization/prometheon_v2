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


def build_image() -> Image:
    """The image every miner's chute references.

    Imported lazily so the SDK stays an optional dependency: a validator scoring
    models never builds anything, and should not need the build toolchain
    installed to run.
    """
    from chutes.image import Image

    image: Any = Image(username=IMAGE_USERNAME, name=IMAGE_NAME, tag=IMAGE_TAG)
    image = image.from_base(f"python:{PYTHON_VERSION}-slim")
    # One command rather than seven: each `run_command` is a layer, and the
    # resolver needs to see the whole set at once to reject an incompatible
    # combination at build time instead of at first inference.
    image = image.run_command("pip install --no-cache-dir " + " ".join(REQUIREMENTS))
    return image


__all__ = ["PYTHON_VERSION", "REQUIREMENTS", "build_image"]
