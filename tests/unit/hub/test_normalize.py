"""The feed's ``normalized_sha256``, which BitFan computes on its side too.

The literal vectors come from running the same rules in Python and in Node, so a
failure here means BitFan's export and the validators no longer agree about what
the same text is. BitFan's own tests check its normaliser against these same
vectors and digests, so change them only together with BitFan.
"""

from __future__ import annotations

import pytest

from prometheon.hub.normalize import normalize_text, normalized_sha256

pytestmark = pytest.mark.unit


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
