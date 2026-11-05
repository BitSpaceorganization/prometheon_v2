"""Near-duplicate removal: how much different content a day really holds.

The data score pays for amount, so the amount has to be real. A hundred copies
of one post, or one post with its punctuation shuffled a hundred ways, is one
piece of evaluation data however it is dressed, and every copy left in would
also be labelled at a validator's expense and tilt the corpus every model is
measured on.

So before anything is labelled, every validator reduces the day's whole corpus,
test and production content together, to the earliest copy of each text. Two
items count as the same when either of these holds:

- their normalised text hashes the same (:func:`normalized_sha256`), or
- the 64-bit SimHashes of their normalised text are within a Hamming distance
  of ``max_hamming`` (:func:`simhash64`).

Everything here is a pure function of the items, so every validator keeps the
same items. The order is ``(created_at, id)``, which is a total order.

**Python's Unicode tables differ between versions**, so a character assigned
after the oldest supported release can normalise differently on two validators.
That affects only text carrying such a character, and only if a near-duplicate
decision hinges on it. It is left to consensus rather than paid for with a
second Unicode database. ``normalized_sha256`` must also agree with BitFan's
export, which is why the rules below use nothing a JavaScript runtime lacks.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from prometheon.errors import ScoringError

SIMHASH_BITS: Final[int] = 64

#: Each feature is three consecutive characters of the normalised text, spaces
#: included. Measured against word and word-pair features on generated posts of
#: 8 to 150 words: three-character features caught the most typos and small edits
#: at distance 3 on short posts, which is what this platform carries, with
#: unrelated texts staying far above the threshold.
_SHINGLE_CHARS: Final[int] = 3

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
    """The text two near-identical posts have in common.

    NFKC, lowercase, every whitespace run becomes one space, punctuation,
    symbols, format and control characters are removed, spaces collapse and trim.
    Punctuation is removed rather than turned into a space so ``h.e.l.l.o`` and
    ``hello`` come out the same.
    """
    lowered = unicodedata.normalize("NFKC", text).lower()
    kept = [
        " " if char in _WHITESPACE else char
        for char in lowered
        if char in _WHITESPACE or not _is_removed(char)
    ]
    return " ".join(part for part in "".join(kept).split(" ") if part)


def normalized_sha256(text: str) -> str:
    """sha256 of the normalised text, lowercase hex. Feeds carry this value too."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def simhash64(normalized: str) -> int:
    """The 64-bit SimHash of already normalised text.

    One feature per three-character shingle, weighted by how often it occurs,
    hashed to the first 8 bytes of its sha256, big-endian. A bit of the result is
    set when the features with that bit set outweigh those without it. A tie
    leaves it clear. Text shorter than a shingle is its own single feature.
    """
    last = max(1, len(normalized) - _SHINGLE_CHARS + 1)
    shingles = Counter(normalized[start : start + _SHINGLE_CHARS] for start in range(last))
    tallies = [0] * SIMHASH_BITS
    for shingle, count in shingles.items():
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        bits = int.from_bytes(digest[:8], "big")
        for position in range(SIMHASH_BITS):
            tallies[position] += count if bits >> position & 1 else -count
    return sum(1 << position for position, tally in enumerate(tallies) if tally > 0)


@dataclass(frozen=True, slots=True)
class CorpusEntry:
    """One item as near-duplicate removal sees it."""

    item_id: str
    #: Microseconds since the Unix epoch. An integer, so ordering never meets a
    #: float.
    created_at_micros: int
    body: str


@dataclass(frozen=True, slots=True)
class DedupeResult:
    #: Ids that survived, earliest first.
    kept: tuple[str, ...]
    #: Each dropped id, and the earlier kept id it duplicates.
    dropped: Mapping[str, str]


def _bands(simhash: int, band_widths: tuple[int, ...]) -> Iterable[tuple[int, int]]:
    """Split a hash into ``(band index, value)`` pairs that together cover it."""
    shift = 0
    for index, width in enumerate(band_widths):
        yield index, simhash >> shift & ((1 << width) - 1)
        shift += width


def _band_widths(max_hamming: int) -> tuple[int, ...]:
    """One more band than differing bits allowed.

    By the pigeonhole principle, two hashes differing in at most ``max_hamming``
    bits agree exactly on at least one of ``max_hamming + 1`` bands, so only
    items sharing a band need comparing. A day can hold tens of thousands of
    items, and comparing all pairs would not finish.
    """
    count = max_hamming + 1
    width = SIMHASH_BITS // count
    return (width,) * (count - 1) + (SIMHASH_BITS - width * (count - 1),)


def drop_near_duplicates(entries: Iterable[CorpusEntry], *, max_hamming: int) -> DedupeResult:
    """Keep the earliest of each group of near-identical items.

    Compared against kept items only. Text A near B near C, where A and C are not
    near each other, keeps A and C: a dropped item never knocks out anything else,
    so nobody can chain small edits from one post towards another's.
    """
    if not 0 <= max_hamming < SIMHASH_BITS:
        raise ScoringError(f"max_hamming must be within [0, {SIMHASH_BITS}), got {max_hamming}")

    ordered = sorted(entries, key=lambda entry: (entry.created_at_micros, entry.item_id))
    widths = _band_widths(max_hamming)

    kept: list[str] = []
    dropped: dict[str, str] = {}
    seen_ids: set[str] = set()
    by_exact_hash: dict[str, str] = {}
    by_band: dict[tuple[int, int], list[tuple[int, str]]] = {}

    for entry in ordered:
        if entry.item_id in seen_ids:
            raise ScoringError(f"duplicate item id in corpus: {entry.item_id!r}")
        seen_ids.add(entry.item_id)

        normalized = normalize_text(entry.body)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        original = by_exact_hash.get(digest)
        if original is not None:
            dropped[entry.item_id] = original
            continue

        simhash = simhash64(normalized)
        bands = list(_bands(simhash, widths))
        match = _nearest_kept(simhash, bands, by_band, max_hamming)
        if match is not None:
            dropped[entry.item_id] = match
            continue

        kept.append(entry.item_id)
        by_exact_hash[digest] = entry.item_id
        for band in bands:
            by_band.setdefault(band, []).append((simhash, entry.item_id))

    return DedupeResult(kept=tuple(kept), dropped=MappingProxyType(dropped))


def _nearest_kept(
    simhash: int,
    bands: list[tuple[int, int]],
    by_band: Mapping[tuple[int, int], list[tuple[int, str]]],
    max_hamming: int,
) -> str | None:
    """The first kept id within range, walking bands and buckets in a fixed order."""
    for band in bands:
        for candidate, item_id in by_band.get(band, ()):
            if (simhash ^ candidate).bit_count() <= max_hamming:
                return item_id
    return None


__all__ = [
    "SIMHASH_BITS",
    "CorpusEntry",
    "DedupeResult",
    "drop_near_duplicates",
    "normalize_text",
    "normalized_sha256",
    "simhash64",
]
