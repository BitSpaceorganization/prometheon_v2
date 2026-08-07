"""The day's evaluation corpus, and the per-miner views of it.

Two properties decide whether a cycle is fair, and both live here.

**Every validator must evaluate the same items in the same order.** Order
matters because a batch is scored as a unit: if two validators split the corpus
differently, a single failed batch costs the two of them different amounts and
their weight vectors diverge for a reason that has nothing to do with the
model. The order is therefore derived from the snapshot's ``content_hash`` by
sorting on a keyed SHA-256 digest rather than by seeding an RNG. A digest sort
is reproducible in any language and any runtime, so a third party recomputing
the cycle gets our order without running our code. ``random.shuffle`` would pin
the subnet to one interpreter's generator.

**A miner is never scored on test content its own users contributed.** Test
items are written by miners' users specifically to violate policy, so a miner
who has seen its own submissions holds answers nobody else holds. The exclusion
is applied when building the per-miner view. That is also why the per-miner
denominator is exposed alongside it: two miners legitimately face different
corpus sizes, and every downstream count has to be taken from the view rather
than from the corpus.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeVar

from prometheon.errors import EvaluationError

#: Mirrors the canonical wrapper's request model. Exceeding either limit makes
#: the wrapper reject the whole request, which would score a batch of good
#: items as incorrect; failing here instead names the offending item.
MAX_ITEMS_PER_REQUEST: Final[int] = 100
MAX_ITEM_ID_CHARS: Final[int] = 128
MAX_CONTENT_CHARS: Final[int] = 32000

#: Domain tag for the ordering digest. Not a signing domain. It exists so this
#: digest can never be confused with, or collide with, another digest taken
#: over the same two strings for a different purpose.
_ORDER_PREFIX: Final[bytes] = b"PROMETHEON_V2_CORPUS_ORDER\n"
_FIELD_SEPARATOR: Final[bytes] = b"\x1f"

_T = TypeVar("_T")


class ItemSource(str, Enum):
    """Where an item came from, which decides how it may be used."""

    #: Miner-attributed content, submitted to violate policy. Positives only,
    #: and excluded from the contributing miner's own view.
    TEST = "test"

    #: Platform production content with no author. Supplies the negatives.
    PRODUCTION = "production"


@dataclass(frozen=True)
class LabelledItem:
    """One item with its ground-truth label, ready to be evaluated."""

    item_id: str
    content: str
    violating: bool
    source: ItemSource
    #: Set only for test items. Production content carries no author, and
    #: enforcing that here means a mislabelled production item can never reach
    #: the self-exclusion path and quietly shrink someone's corpus.
    contributor_hotkey: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise EvaluationError("item_id must not be empty")
        if len(self.item_id) > MAX_ITEM_ID_CHARS:
            raise EvaluationError(
                f"item_id {self.item_id[:32]!r}... is {len(self.item_id)} characters, "
                f"over the {MAX_ITEM_ID_CHARS} the canonical wrapper accepts"
            )
        if len(self.content) > MAX_CONTENT_CHARS:
            raise EvaluationError(
                f"item {self.item_id!r} has {len(self.content)} characters of content, "
                f"over the {MAX_CONTENT_CHARS} the canonical wrapper accepts"
            )
        if self.source is ItemSource.TEST and not self.contributor_hotkey:
            raise EvaluationError(
                f"test item {self.item_id!r} has no contributor hotkey, so it cannot be "
                "excluded from its contributor's own evaluation"
            )
        if self.source is ItemSource.PRODUCTION and self.contributor_hotkey is not None:
            raise EvaluationError(
                f"production item {self.item_id!r} carries a contributor hotkey; "
                "production content is unattributed"
            )


@dataclass(frozen=True)
class MinerCorpusCounts:
    """The denominators a caller needs for one miner's view."""

    hotkey: str
    total: int
    positives: int
    negatives: int
    #: Test items dropped because this miner's own users contributed them.
    excluded: int


def corpus_order_key(content_hash: str, item_id: str) -> bytes:
    """The sort key that fixes an item's position in the evaluation order."""
    return hashlib.sha256(
        _ORDER_PREFIX + content_hash.encode("utf-8") + _FIELD_SEPARATOR + item_id.encode("utf-8")
    ).digest()


@dataclass(frozen=True)
class EvaluationCorpus:
    """The combined corpus for one cycle, already in evaluation order."""

    content_hash: str
    items: tuple[LabelledItem, ...]

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def positives(self) -> int:
        return sum(1 for item in self.items if item.violating)

    @property
    def negatives(self) -> int:
        return sum(1 for item in self.items if not item.violating)

    @property
    def test_items(self) -> int:
        return sum(1 for item in self.items if item.source is ItemSource.TEST)

    @property
    def production_items(self) -> int:
        return sum(1 for item in self.items if item.source is ItemSource.PRODUCTION)

    def for_miner(self, hotkey: str) -> tuple[LabelledItem, ...]:
        """This miner's view: everything except test items it contributed.

        Relative order is preserved, so removing items cannot reshuffle what is
        left and every miner's batches line up with the same global sequence.
        """
        if not hotkey:
            raise EvaluationError(
                "a miner hotkey is required to build a corpus view; an empty hotkey "
                "would exclude nothing and score a miner on its own test items"
            )
        return tuple(item for item in self.items if item.contributor_hotkey != hotkey)

    def counts_for_miner(self, hotkey: str) -> MinerCorpusCounts:
        view = self.for_miner(hotkey)
        positives = sum(1 for item in view if item.violating)
        return MinerCorpusCounts(
            hotkey=hotkey,
            total=len(view),
            positives=positives,
            negatives=len(view) - positives,
            excluded=len(self.items) - len(view),
        )


def build_corpus(
    *,
    content_hash: str,
    test_items: Iterable[LabelledItem],
    production_items: Iterable[LabelledItem],
) -> EvaluationCorpus:
    """Combine the labelled halves of a snapshot into one ordered corpus.

    Non-violating test items are rejected rather than dropped. Labelling is
    contracted to have discarded them already, and silently dropping them here
    would hide a labelling regression behind a corpus that merely looks smaller
    than usual.
    """
    if not content_hash.strip():
        raise EvaluationError(
            "content_hash must not be empty: it is the seed every validator "
            "derives the evaluation order from"
        )

    collected: list[LabelledItem] = []
    seen: set[str] = set()

    for item in test_items:
        if item.source is not ItemSource.TEST:
            raise EvaluationError(
                f"item {item.item_id!r} was supplied as test content but is marked "
                f"{item.source.value}"
            )
        if not item.violating:
            raise EvaluationError(
                f"test item {item.item_id!r} is labelled non-violating; test content "
                "supplies positives only and must be filtered during labelling"
            )
        _add(collected, seen, item)

    for item in production_items:
        if item.source is not ItemSource.PRODUCTION:
            raise EvaluationError(
                f"item {item.item_id!r} was supplied as production content but is marked "
                f"{item.source.value}"
            )
        _add(collected, seen, item)

    ordered = sorted(
        collected,
        key=lambda item: (corpus_order_key(content_hash, item.item_id), item.item_id),
    )
    return EvaluationCorpus(content_hash=content_hash, items=tuple(ordered))


def _add(collected: list[LabelledItem], seen: set[str], item: LabelledItem) -> None:
    """Append an item, refusing a duplicate id.

    Ids are how a model's verdicts are matched back to items. Two items sharing
    one id make that matching ambiguous, so the corpus refuses to contain them.
    """
    if item.item_id in seen:
        raise EvaluationError(f"duplicate item id in corpus: {item.item_id!r}")
    seen.add(item.item_id)
    collected.append(item)


def iter_batches(items: Sequence[_T], size: int) -> Iterator[tuple[_T, ...]]:
    """Yield fixed-size batches, the last one short if it has to be."""
    if size < 1 or size > MAX_ITEMS_PER_REQUEST:
        raise EvaluationError(
            f"batch size must be between 1 and {MAX_ITEMS_PER_REQUEST}, got {size}"
        )
    for start in range(0, len(items), size):
        yield tuple(items[start : start + size])


__all__ = [
    "MAX_CONTENT_CHARS",
    "MAX_ITEMS_PER_REQUEST",
    "MAX_ITEM_ID_CHARS",
    "EvaluationCorpus",
    "ItemSource",
    "LabelledItem",
    "MinerCorpusCounts",
    "build_corpus",
    "corpus_order_key",
    "iter_batches",
]
