"""Corpus construction, deterministic ordering, and self-exclusion."""

from __future__ import annotations

import pytest

from prometheon.errors import EvaluationError
from prometheon.evaluation.corpus import (
    MAX_CONTENT_CHARS,
    MAX_ITEM_ID_CHARS,
    EvaluationCorpus,
    ItemSource,
    LabelledItem,
    build_corpus,
    corpus_order_key,
    iter_batches,
)

pytestmark = pytest.mark.unit

MINER_A = "5FA0miner_a"
MINER_B = "5FB0miner_b"


def make_test_item(item_id: str, hotkey: str = MINER_A, *, violating: bool = True) -> LabelledItem:
    return LabelledItem(
        item_id=item_id,
        content=f"test content {item_id}",
        violating=violating,
        source=ItemSource.TEST,
        contributor_hotkey=hotkey,
    )


def production_item(item_id: str, *, violating: bool = False) -> LabelledItem:
    return LabelledItem(
        item_id=item_id,
        content=f"production content {item_id}",
        violating=violating,
        source=ItemSource.PRODUCTION,
    )


def build(
    content_hash: str = "a" * 64,
    *,
    tests: list[LabelledItem] | None = None,
    production: list[LabelledItem] | None = None,
) -> EvaluationCorpus:
    default_tests = [make_test_item("t1"), make_test_item("t2", MINER_B)]
    default_production = [production_item("p1"), production_item("p2", violating=True)]
    return build_corpus(
        content_hash=content_hash,
        test_items=tests if tests is not None else default_tests,
        production_items=production if production is not None else default_production,
    )


# --- ordering ---------------------------------------------------------------


def test_order_is_identical_for_the_same_content_hash() -> None:
    tests = [make_test_item(f"t{i}") for i in range(20)]
    production = [production_item(f"p{i}") for i in range(30)]

    first = build(tests=list(tests), production=list(production))
    # A different input order must not survive into the evaluation order:
    # validators receive snapshot rows in whatever order the DB layer returns.
    second = build(tests=list(reversed(tests)), production=list(reversed(production)))

    assert [item.item_id for item in first.items] == [item.item_id for item in second.items]


def test_order_changes_with_the_content_hash() -> None:
    tests = [make_test_item(f"t{i}") for i in range(20)]
    production = [production_item(f"p{i}") for i in range(30)]

    day_one = build("a" * 64, tests=list(tests), production=list(production))
    day_two = build("b" * 64, tests=list(tests), production=list(production))

    assert [i.item_id for i in day_one.items] != [i.item_id for i in day_two.items]


def test_order_is_a_permutation_of_the_inputs() -> None:
    tests = [make_test_item(f"t{i}") for i in range(7)]
    production = [production_item(f"p{i}") for i in range(11)]
    corpus = build(tests=tests, production=production)

    assert sorted(item.item_id for item in corpus.items) == sorted(
        item.item_id for item in tests + production
    )
    assert corpus.size == 18


def test_order_is_not_the_sorted_input_order() -> None:
    """A shuffle that happens to be sorted order would be no shuffle at all."""
    items = [production_item(f"p{i:03d}") for i in range(40)]
    corpus = build(production=items, tests=[])

    ids = [item.item_id for item in corpus.items]
    assert ids != sorted(ids)


def test_order_key_is_pinned_to_the_documented_derivation() -> None:
    """Third parties recompute the order from this derivation, not from us."""
    import hashlib

    expected = hashlib.sha256(
        b"PROMETHEON_V2_CORPUS_ORDER\n" + b"cafe" + b"\x1f" + b"item-7"
    ).digest()
    assert corpus_order_key("cafe", "item-7") == expected


def test_order_key_separates_the_two_fields() -> None:
    """Without a separator, ('ab','c') and ('a','bc') would collide."""
    assert corpus_order_key("ab", "c") != corpus_order_key("a", "bc")


# --- self-exclusion ---------------------------------------------------------


def test_a_miner_is_not_scored_on_its_own_test_items() -> None:
    corpus = build(
        tests=[make_test_item("mine", MINER_A), make_test_item("theirs", MINER_B)],
        production=[production_item("p1")],
    )

    view_ids = {item.item_id for item in corpus.for_miner(MINER_A)}
    assert view_ids == {"theirs", "p1"}


def test_production_items_are_never_excluded() -> None:
    corpus = build(tests=[make_test_item("mine", MINER_A)], production=[production_item("p1")])

    for hotkey in (MINER_A, MINER_B, "5FC0unknown"):
        assert "p1" in {item.item_id for item in corpus.for_miner(hotkey)}


def test_a_miner_with_no_contributions_sees_the_whole_corpus() -> None:
    corpus = build(tests=[make_test_item("t1", MINER_A)], production=[production_item("p1")])

    assert len(corpus.for_miner("5FC0stranger")) == corpus.size


def test_exclusion_preserves_the_global_order() -> None:
    tests = [make_test_item(f"t{i}", MINER_A if i % 2 else MINER_B) for i in range(20)]
    corpus = build(tests=tests, production=[production_item(f"p{i}") for i in range(20)])

    global_order = [item.item_id for item in corpus.items]
    view_order = [item.item_id for item in corpus.for_miner(MINER_A)]

    assert view_order == [item_id for item_id in global_order if item_id in set(view_order)]


def test_an_empty_hotkey_is_refused_rather_than_excluding_nothing() -> None:
    corpus = build()
    with pytest.raises(EvaluationError, match="hotkey"):
        corpus.for_miner("")


# --- counts -----------------------------------------------------------------


def test_counts_report_the_per_miner_denominator() -> None:
    corpus = build(
        tests=[
            make_test_item("t1", MINER_A),
            make_test_item("t2", MINER_A),
            make_test_item("t3", MINER_B),
        ],
        production=[
            production_item("p1"),
            production_item("p2"),
            production_item("p3", violating=True),
        ],
    )

    counts = corpus.counts_for_miner(MINER_A)
    assert counts.total == 4
    assert counts.excluded == 2
    assert counts.positives == 2  # t3 plus the violating production item
    assert counts.negatives == 2
    assert counts.total == counts.positives + counts.negatives

    other = corpus.counts_for_miner(MINER_B)
    assert other.total == 5
    assert other.excluded == 1


def test_corpus_level_counts() -> None:
    corpus = build(
        tests=[make_test_item("t1")],
        production=[production_item("p1"), production_item("p2", violating=True)],
    )

    assert (corpus.positives, corpus.negatives) == (2, 1)
    assert (corpus.test_items, corpus.production_items) == (1, 2)


# --- validation -------------------------------------------------------------


def test_a_non_violating_test_item_is_kept_as_a_negative() -> None:
    """Test content carries its verdict, negatives included.

    Production used to be the only source of negatives, which left the corpus
    at the mercy of how much production content a day happened to have. A test
    item the labeller judged non-violating is a real negative and is admitted
    as one; it still costs its author, because it counts in ``S`` and not in
    ``V``.
    """
    corpus = build(
        tests=[make_test_item("t1"), make_test_item("t2", violating=False)],
        production=[production_item("p1")],
    )

    assert corpus.size == 3
    assert (corpus.positives, corpus.negatives) == (1, 2)
    assert corpus.test_items == 2
    verdicts = {item.item_id: item.violating for item in corpus.items}
    assert verdicts == {"t1": True, "t2": False, "p1": False}


def test_sources_must_match_the_half_they_were_supplied_in() -> None:
    with pytest.raises(EvaluationError, match="production content but is marked"):
        build(production=[make_test_item("t1")])
    with pytest.raises(EvaluationError, match="test content but is marked"):
        build(tests=[production_item("p1", violating=True)])


def test_duplicate_ids_are_refused() -> None:
    with pytest.raises(EvaluationError, match="duplicate item id"):
        build(tests=[make_test_item("dupe")], production=[production_item("dupe")])


def test_an_empty_content_hash_is_refused() -> None:
    with pytest.raises(EvaluationError, match="content_hash"):
        build("   ")


def test_a_test_item_needs_a_contributor() -> None:
    with pytest.raises(EvaluationError, match="contributor hotkey"):
        LabelledItem(item_id="t1", content="x", violating=True, source=ItemSource.TEST)


def test_production_content_may_not_carry_an_author() -> None:
    with pytest.raises(EvaluationError, match="unattributed"):
        LabelledItem(
            item_id="p1",
            content="x",
            violating=False,
            source=ItemSource.PRODUCTION,
            contributor_hotkey=MINER_A,
        )


def test_items_over_the_engine_limits_are_named_here() -> None:
    with pytest.raises(EvaluationError, match="evaluation engine accepts"):
        production_item("x" * (MAX_ITEM_ID_CHARS + 1))
    with pytest.raises(EvaluationError, match="evaluation engine accepts"):
        LabelledItem(
            item_id="p1",
            content="x" * (MAX_CONTENT_CHARS + 1),
            violating=False,
            source=ItemSource.PRODUCTION,
        )


def test_an_empty_item_id_is_refused() -> None:
    with pytest.raises(EvaluationError, match="item_id"):
        production_item("")


def test_an_empty_corpus_is_allowed_and_reports_zero() -> None:
    corpus = build(tests=[], production=[])
    assert corpus.size == 0
    assert corpus.counts_for_miner(MINER_A).total == 0


# --- batching ---------------------------------------------------------------


def test_batches_are_full_until_the_last_one() -> None:
    items = [production_item(f"p{i}") for i in range(250)]
    batches = list(iter_batches(items, 100))

    assert [len(batch) for batch in batches] == [100, 100, 50]
    assert [item.item_id for batch in batches for item in batch] == [item.item_id for item in items]


def test_an_exact_multiple_yields_no_empty_tail() -> None:
    items = [production_item(f"p{i}") for i in range(200)]
    assert [len(batch) for batch in iter_batches(items, 100)] == [100, 100]


def test_no_items_means_no_batches() -> None:
    assert list(iter_batches([], 100)) == []


@pytest.mark.parametrize("size", [0, -1, 101])
def test_batch_size_is_bounded_by_the_engine_contract(size: int) -> None:
    with pytest.raises(EvaluationError, match="batch size"):
        list(iter_batches([production_item("p1")], size))
