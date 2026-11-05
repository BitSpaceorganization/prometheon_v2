"""Near-duplicate removal.

The literal vectors come from running the same rules in Python and in Node, so a
failure here means BitFan's export and the validators no longer agree about what
the same text is.
"""

from __future__ import annotations

import pytest

from prometheon.errors import ScoringError
from prometheon.scoring.dedupe import (
    CorpusEntry,
    drop_near_duplicates,
    normalize_text,
    normalized_sha256,
    simhash64,
)

pytestmark = pytest.mark.unit

MAX_HAMMING = 3


def _entry(item_id: str, body: str, at: int = 0) -> CorpusEntry:
    return CorpusEntry(item_id=item_id, created_at_micros=at, body=body)


class TestNormalisation:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Hello,   WORLD!!!", "hello world"),
            ("  Tabs\tand\nnewlines\r\n here  ", "tabs and newlines here"),
            ("h.e.l.l.o", "hello"),
            ("don't stop-me now", "dont stopme now"),
            ("\uff26\uff55\uff4c\uff4c\u3000\uff37\uff49\uff44\uff54\uff48", "full width"),
            ("zero\u200bwidth\u200djoin\ufeffer", "zerowidthjoiner"),
            ("emoji \U0001f600\U0001f389 here", "emoji here"),
            ("$100 + \u20ac50 = ~150", "100 50 150"),
            (
                "\u03a3\u0391\u03a3 \u039f\u0394\u039f\u03a3",
                "\u03c3\u03b1\u03c2 \u03bf\u03b4\u03bf\u03c2",
            ),
            ("\u0130stanbul", "i\u0307stanbul"),
            ("Stra\u00dfe", "stra\u00dfe"),
            (
                "\u041f\u0440\u0438\u0432\u0435\u0442, \u041c\u0418\u0420!",
                "\u043f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440",
            ),
            ("!!!", ""),
            ("", ""),
            ("a - b", "a b"),
            ("caf\u00e9 cafe\u0301", "caf\u00e9 caf\u00e9"),
            ("\x01ctrl\x1fchars\x7f", "ctrlchars"),
            ("\U0001f5d1\ufe0f trash", "trash"),
            ("NEL\x85here", "nel here"),
            ("LS\u2028PS\u2029end", "ls ps end"),
        ],
    )
    def test_the_rules_every_implementation_must_match(self, text: str, expected: str) -> None:
        assert normalize_text(text) == expected

    @pytest.mark.parametrize(
        ("text", "digest"),
        [
            (
                "Hello,   WORLD!!!",
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            ),
            ("h.e.l.l.o", "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"),
            ("!!!", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        ],
    )
    def test_the_published_digests(self, text: str, digest: str) -> None:
        assert normalized_sha256(text) == digest


class TestSimHash:
    def test_it_is_a_pinned_function_of_the_text(self) -> None:
        """Every validator must compute the same hash, on any Python version."""
        assert simhash64("hello world") == 0x92C886D86A59BB7C

    def test_it_fits_in_64_bits(self) -> None:
        assert 0 <= simhash64("x" * 5000) < 1 << 64

    def test_text_shorter_than_a_shingle_is_still_hashed(self) -> None:
        assert simhash64("ab") != simhash64("cd")

    def test_a_one_character_change_in_a_long_post_stays_close(self) -> None:
        post = " ".join(f"word{index}" for index in range(120))
        edited = post.replace("word60", "wxrd60")
        assert (simhash64(post) ^ simhash64(edited)).bit_count() <= MAX_HAMMING

    def test_unrelated_posts_are_far_apart(self) -> None:
        first = "please review the rules about harassment before posting again"
        second = "our new phone arrived yesterday and the battery lasts two days"
        assert (simhash64(first) ^ simhash64(second)).bit_count() > 3 * MAX_HAMMING


class TestDropNearDuplicates:
    def test_exact_copies_keep_only_the_earliest(self) -> None:
        result = drop_near_duplicates(
            [_entry("late", "same text", 20), _entry("early", "same text", 10)],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("early",)
        assert result.dropped == {"late": "early"}

    def test_case_whitespace_and_punctuation_do_not_make_a_new_item(self) -> None:
        result = drop_near_duplicates(
            [
                _entry("a", "You are all idiots", 1),
                _entry("b", "YOU ARE ALL IDIOTS!!!", 2),
                _entry("c", "you   are\tall, idiots.", 3),
            ],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("a",)
        assert set(result.dropped) == {"b", "c"}

    def test_a_small_edit_to_a_long_post_is_a_duplicate(self) -> None:
        post = " ".join(f"word{index}" for index in range(120))
        result = drop_near_duplicates(
            [_entry("a", post, 1), _entry("b", post.replace("word60", "wxrd60"), 2)],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("a",)
        assert result.dropped == {"b": "a"}

    def test_different_posts_both_stay(self) -> None:
        result = drop_near_duplicates(
            [
                _entry("a", "please review the rules about harassment before posting", 1),
                _entry("b", "our new phone arrived yesterday and the battery is great", 2),
            ],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("a", "b")
        assert not result.dropped

    def test_a_tie_on_time_goes_to_the_lower_id(self) -> None:
        result = drop_near_duplicates(
            [_entry("b", "same text", 5), _entry("a", "same text", 5)],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("a",)

    def test_input_order_cannot_change_the_outcome(self) -> None:
        entries = [_entry(f"i{index}", f"text number {index % 3}", index) for index in range(9)]
        forward = drop_near_duplicates(entries, max_hamming=MAX_HAMMING)
        backward = drop_near_duplicates(reversed(entries), max_hamming=MAX_HAMMING)
        assert forward.kept == backward.kept
        assert forward.dropped == backward.dropped

    def test_a_dropped_item_cannot_knock_out_a_later_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only kept items are compared.

        A is kept. B is within range of A, so it is dropped. C is within range
        of B but not of A, and B is gone, so C stays. Comparing against every
        item seen would let a chain of small edits walk from one post to an
        unrelated one and take everything along the way down with it.
        """
        hashes = {"a": 0b0000, "b": 0b0111, "c": 0b1111}
        assert (hashes["a"] ^ hashes["b"]).bit_count() == MAX_HAMMING
        assert (hashes["b"] ^ hashes["c"]).bit_count() == 1
        assert (hashes["a"] ^ hashes["c"]).bit_count() == MAX_HAMMING + 1
        monkeypatch.setattr("prometheon.scoring.dedupe.simhash64", lambda text: hashes[text])

        result = drop_near_duplicates(
            [_entry("1", "a", 1), _entry("2", "b", 2), _entry("3", "c", 3)],
            max_hamming=MAX_HAMMING,
        )
        assert result.kept == ("1", "3")
        assert result.dropped == {"2": "1"}

    def test_a_repeated_id_is_refused(self) -> None:
        with pytest.raises(ScoringError, match="duplicate item id"):
            drop_near_duplicates(
                [_entry("a", "one", 1), _entry("a", "two", 2)], max_hamming=MAX_HAMMING
            )

    @pytest.mark.parametrize("bad", [-1, 64])
    def test_a_nonsense_distance_is_refused(self, bad: int) -> None:
        with pytest.raises(ScoringError, match="max_hamming"):
            drop_near_duplicates([], max_hamming=bad)

    def test_zero_distance_still_finds_identical_hashes(self) -> None:
        result = drop_near_duplicates(
            [_entry("a", "same text", 1), _entry("b", "same text!", 2)], max_hamming=0
        )
        assert result.kept == ("a",)

    def test_a_flood_of_near_copies_reduces_to_one(self) -> None:
        """Flood resistance: one post a thousand ways is one item."""
        post = " ".join(f"word{index}" for index in range(200))
        entries = [_entry("original", post, 0)]
        for index in range(1000):
            noise = "".join(".,!?;:"[int(digit) % 6] for digit in str(index))
            entries.append(_entry(f"copy{index}", f"{post}{noise}", index + 1))
        result = drop_near_duplicates(entries, max_hamming=MAX_HAMMING)
        assert result.kept == ("original",)
        assert len(result.dropped) == 1000
