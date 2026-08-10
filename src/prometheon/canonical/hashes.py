"""The wrapper hashes validators accept this cycle.

Validators accept a **set**, never a single hash. A bump with no overlap
invalidates every deployment in the subnet at the same instant, which is a
subnet-wide outage with extra steps.

Migration, as documented in ``docs/canonical-wrapper.md``:

- Standard bump: the new hash is published here seven days before the old one
  is removed. Both score normally in between, with no penalty for either.
- Emergency bump: 48 hours, reserved for a defect in the wrapper itself, such
  as a prompt-injection escape or a verdict-forging path. Some miners will drop
  out, and that is the accepted cost of closing a live hole.

A validator that does not pull the repository will reject deployments the rest
of the field accepts, diverge, and lose vtrust. That is the same enforcement
that governs policy updates, and it needs no additional machinery.
"""

from __future__ import annotations

import uuid
from typing import Final

from prometheon.canonical.integrity import canonical_wrapper_hash

WRAPPER_VERSION: Final[str] = "prometheon-moderation/1"

#: Hashes retired but still accepted during a migration window. Empty outside
#: one. Entries are removed on the announced cutoff date, never earlier.
LEGACY_WRAPPER_HASHES: Final[frozenset[str]] = frozenset()


def accepted_wrapper_hashes() -> frozenset[str]:
    """Current canonical hash plus any inside an open migration window."""
    return frozenset({canonical_wrapper_hash()}) | LEGACY_WRAPPER_HASHES


#: Convenience for callers that want the set without a function call. Computed
#: once at import; the canonical hash cannot change at runtime.
ACCEPTED_WRAPPER_HASHES: Final[frozenset[str]] = accepted_wrapper_hashes()

__all__ = [
    "ACCEPTED_WRAPPER_HASHES",
    "CHUTE_NAME_PREFIX",
    "IMAGE_NAME",
    "IMAGE_TAG",
    "IMAGE_USERNAME",
    "LEGACY_IMAGE_IDS",
    "LEGACY_WRAPPER_HASHES",
    "WRAPPER_VERSION",
    "accepted_image_ids",
    "accepted_wrapper_hashes",
    "chute_id_for",
    "image_id_for",
]


# ---------------------------------------------------------------------------
# The subnet's published image
# ---------------------------------------------------------------------------

#: The image every miner's chute must reference, as ``username/name:tag``.
#:
#: The subnet's own Chutes account. Miners build nothing: they reference this
#: image by id, so the runtime their weights load inside is one the subnet
#: published and every validator can name.
#:
#: These three strings *are* the id — ``image_id_for`` is a pure function of
#: them — so changing any of them is an image migration, and
#: ``LEGACY_IMAGE_IDS`` is how it gets an overlap window.
IMAGE_USERNAME: Final[str] = "coolaisaworld"
IMAGE_NAME: Final[str] = "prometheon-moderation"
IMAGE_TAG: Final[str] = "1"

#: Image ids retired but still accepted during a migration window. Empty outside
#: one. Rebuilding the image — a torch bump, a CUDA bump — is more frequent than
#: changing the script, so it needs the overlap more, not less.
LEGACY_IMAGE_IDS: Final[frozenset[str]] = frozenset()


def image_id_for(username: str, name: str, tag: str) -> str:
    """The id Chutes derives for an image, computed rather than looked up.

    ``uuid5(NAMESPACE_OID, f"{username}/{name}:{tag}".lower())`` — a pure
    function of the three strings, so the expected id is knowable before the
    image is ever built, and ``chutes build`` producing a different one is
    itself a detectable fault.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{username}/{name}:{tag}".lower()))


def accepted_image_ids() -> frozenset[str]:
    """Ids a validator will accept on a deployed chute this cycle.

    Empty while the image is unconfigured. The registry treats an empty set as
    "this check cannot be performed" and refuses every deployment, which is the
    safe direction: accepting anything would mean accepting a miner-built image,
    and the Chutes API exposes no image contents to tell them apart.
    """
    if not (IMAGE_USERNAME and IMAGE_NAME and IMAGE_TAG):
        return frozenset()
    return frozenset({image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG)}) | LEGACY_IMAGE_IDS


#: How the canonical script names a chute: the prefix plus the miner's hotkey.
#: Kept beside the id derivation so the two cannot drift from the template.
CHUTE_NAME_PREFIX: Final[str] = "prometheon-"


def chute_id_for(chutes_user: str, hotkey: str) -> str:
    """The id Chutes will give this miner's deployment, computed in advance.

    ``uuid5(NAMESPACE_OID, f"{username}::chute::{name}")``, and the name is
    fixed by the template as ``prometheon-<hotkey>``. Two things follow: a miner
    can commit the id before the chute exists — which they had no way to do when
    the name was theirs to choose — and a validator can check that the chute it
    fetched is the one the commitment named, rather than taking the field on
    trust.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_OID, f"{chutes_user}::chute::{CHUTE_NAME_PREFIX}{hotkey}"))
