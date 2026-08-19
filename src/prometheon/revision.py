"""What counts as a commitment's revision.

One rule, used by the chain codec, the registry and the DB client, which is why
it lives above all three rather than inside any of them.

This used to sit in ``prometheon.canonical`` beside the wrapper miners
deployed. That package existed to pin *code* -- one engine, hash-checked, so
every model on the subnet ran identical inference. Validators run the engine
themselves now, so there is no miner-supplied code left to pin and nothing
canonical to guard. The revision rule outlived it, because it never had
anything to do with the wrapper.
"""

from __future__ import annotations

import re
from typing import Final

from prometheon.errors import RevisionFormatError

_HF_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


def is_valid_revision(revision: str) -> bool:
    """True only for a 40-character lowercase hex commit SHA.

    Branches and tags are rejected: a moving reference is not a commitment, and
    accepting one lets a miner change the model under a revision that was
    already verified.
    """
    return bool(_HF_SHA_RE.match(revision or ""))


def require_valid_revision(revision: str) -> str:
    if not is_valid_revision(revision):
        raise RevisionFormatError(
            f"revision must be a 40-character lowercase hex SHA, got {revision!r}"
        )
    return revision
