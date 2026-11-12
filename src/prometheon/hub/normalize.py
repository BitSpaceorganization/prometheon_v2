"""The feed's ``normalized_sha256``: the text two near-identical posts share.

BitFan writes this value on every test line, and refuses a post whose
normalised text is too close to one already written. The validator no longer
removes near-duplicates itself, but the feed format still carries the field,
so a line without a well-formed value is rejected, and a value that disagrees
with this validator's own computation is counted in the operator's log. That
count is how a drift between BitFan's normaliser and this one becomes visible.

The rules use nothing a JavaScript runtime lacks, because BitFan computes the
same value on its side. ``docs/hub-contract.md`` states them, with vectors.

**Python's Unicode tables differ between versions**, so a character assigned
after the oldest supported release can normalise differently on two machines.
That changes only the logged mismatch count, never a score.
"""

from __future__ import annotations

import hashlib
import unicodedata
from typing import Final

#: Characters treated as whitespace. Listed out, rather than `str.isspace`, so a
#: JavaScript implementation can match it byte for byte: the two runtimes
#: disagree about U+001C-U+001F, U+0085 and U+FEFF.
_WHITESPACE: Final[frozenset[str]] = frozenset(
    chr(code)
    for code in (
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x20,
        0x85,
        0xA0,
        0x1680,
        *range(0x2000, 0x200B),
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    )
)


def _is_removed(char: str) -> bool:
    """Punctuation, symbols, invisible format characters and controls.

    Format characters and variation selectors are here because they are how a
    copy stays visually identical while its bytes change: a zero-width space
    between two letters survives every other rule on this page.
    """
    category = unicodedata.category(char)
    return category[0] in "PS" or category in ("Cf", "Cc") or 0xFE00 <= ord(char) <= 0xFE0F


def normalize_text(text: str) -> str:
    """NFKC, lowercase, whitespace runs to one space, punctuation and symbols out.

    Punctuation is removed rather than turned into a space, so ``h.e.l.l.o``
    and ``hello`` come out the same.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    kept = [
        " " if char in _WHITESPACE else char
        for char in lowered
        if char in _WHITESPACE or not _is_removed(char)
    ]
    return " ".join(part for part in "".join(kept).split(" ") if part)


def normalized_sha256(text: str) -> str:
    """sha256 of the normalised text, lowercase hex, as the feed carries it."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


__all__ = ["normalize_text", "normalized_sha256"]
