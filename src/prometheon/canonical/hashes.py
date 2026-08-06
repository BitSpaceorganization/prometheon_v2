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
    "LEGACY_WRAPPER_HASHES",
    "WRAPPER_VERSION",
    "accepted_wrapper_hashes",
]
