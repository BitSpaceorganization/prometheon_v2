"""Feed files: checked against the epoch, then line by line.

Every rule here is one a line can fail without costing anyone else their day.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from typing import Any

import pytest
from tests.unit.hub.factories import (
    DAY,
    MINER_A,
    MINER_B,
    feed_test_line,
    production_line,
    standard_miners,
)

from prometheon.errors import FeedFileError
from prometheon.hub.fake import gzip_lines
from prometheon.hub.feed import parse_feed, parse_timestamp_micros, read_records
from prometheon.hub.models import EpochFile, EpochMiner

pytestmark = pytest.mark.unit

MINERS = {row["hotkey"]: EpochMiner.model_validate(row) for row in standard_miners()}


def _records(lines: list[Any]) -> list[bytes]:
    # Rendered with ASCII escapes, so a lone surrogate can travel the way JSON
    # allows it to: as `\ud800`, which only becomes a problem once parsed.
    rendered = [line if isinstance(line, bytes) else json.dumps(line).encode() for line in lines]
    data = gzip_lines(rendered)
    expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=len(lines))
    return read_records(data, name="test", expected=expected)


def _parse(tests: list[Any], productions: list[Any] | None = None) -> Any:
    return parse_feed(
        test_records=_records(tests),
        production_records=_records(productions or []),
        day=DAY,
        miners=MINERS,
    )


def _good(item_id: str = "t1", **overrides: Any) -> dict[str, Any]:
    """A conforming line, then the overrides applied on top, hash left as it was."""
    line = feed_test_line(item_id, MINER_A.ss58_address, "a real test post")
    line.update(overrides)
    return line


class TestTheFileMustBeTheSignedOne:
    def test_a_hash_mismatch_stops_the_day(self) -> None:
        data = gzip_lines([_good()])
        with pytest.raises(FeedFileError, match="hashes to"):
            read_records(data, name="test", expected=EpochFile(sha256="0" * 64, count=1))

    def test_a_line_count_mismatch_stops_the_day(self) -> None:
        data = gzip_lines([_good()])
        expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=2)
        with pytest.raises(FeedFileError, match="holds 1 lines"):
            read_records(data, name="test", expected=expected)

    def test_not_gzip_stops_the_day(self) -> None:
        data = b"not gzip at all"
        expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=0)
        with pytest.raises(FeedFileError, match="not valid gzip"):
            read_records(data, name="test", expected=expected)

    def test_a_truncated_archive_stops_the_day(self) -> None:
        data = gzip_lines([_good()])[:-12]
        expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=1)
        with pytest.raises(FeedFileError):
            read_records(data, name="test", expected=expected)

    def test_a_multi_member_archive_is_read_whole(self) -> None:
        first = json.dumps(_good("t1")).encode() + b"\n"
        second = json.dumps(_good("t2")).encode() + b"\n"
        data = gzip.compress(first) + gzip.compress(second)
        expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=2)
        assert len(read_records(data, name="test", expected=expected)) == 2

    def test_a_decompression_bomb_stops_at_the_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("prometheon.hub.feed.MAX_DECOMPRESSED_BYTES", 1000)
        data = gzip.compress(b"x" * 100_000)
        expected = EpochFile(sha256=hashlib.sha256(data).hexdigest(), count=1)
        with pytest.raises(FeedFileError, match="ceiling"):
            read_records(data, name="test", expected=expected)

    def test_an_empty_file_has_no_lines(self) -> None:
        assert _records([]) == []


class TestTestLines:
    def test_a_conforming_line_is_accepted(self) -> None:
        feed = _parse([_good()])
        (line,) = feed.test
        assert line.miner_hotkey == MINER_A.ss58_address
        assert line.claimed_violating is True
        assert not feed.rejected
        assert feed.hash_mismatches == 0

    @pytest.mark.parametrize(
        ("overrides", "reason"),
        [
            ({"verified": None}, "no verified marker"),
            ({"verified": False}, "no verified marker"),
            ({"verified": "true"}, "no verified marker"),
            ({"author": ""}, "author"),
            ({"author": None}, "author"),
            ({"claimed_violating": "yes"}, "claimed_violating"),
            ({"claimed_violating": 1}, "claimed_violating"),
            ({"normalized_sha256": None}, "normalized_sha256"),
            ({"normalized_sha256": "ABC"}, "normalized_sha256"),
            ({"miner_hotkey": "not-a-hotkey"}, "SS58"),
            ({"created_at": "yesterday"}, "RFC 3339"),
            ({"created_at": "2026-11-06T00:00:00Z"}, "outside"),
            ({"created_at": "2026-11-04T23:59:59.999999Z"}, "outside"),
            ({"body": ""}, "body"),
            ({"body": "x" * 32_001}, "body"),
            ({"body": "\ud800 lone surrogate"}, "body"),
            ({"id": "x" * 129}, "id"),
        ],
    )
    def test_a_line_missing_what_the_spec_requires_is_rejected(
        self, overrides: dict[str, Any], reason: str
    ) -> None:
        line = _good(**overrides)
        if overrides.get("verified", True) is None:
            del line["verified"]
        feed = _parse([_good("t0"), line])
        assert [kept.item_id for kept in feed.test] == ["t0"]
        (rejected,) = feed.rejected
        assert rejected.file == "test"
        assert rejected.line_number == 2
        assert reason in rejected.reason

    def test_a_line_for_a_hotkey_not_in_the_epoch_is_rejected(self) -> None:
        stranger = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
        feed = _parse([_good("t0"), feed_test_line("t1", stranger, "a post")])
        assert [rejected.reason for rejected in feed.rejected] == [
            "miner_hotkey is not in the epoch's miner list"
        ]

    def test_a_rejection_never_echoes_content(self) -> None:
        feed = _parse([_good("t0"), _good("t1", verified=False, body="SECRET TEXT")])
        assert "SECRET" not in repr(feed.rejected)

    def test_a_wrong_normalised_hash_is_counted_not_rejected(self) -> None:
        """BitFan's value is checked, but a drift must not wipe out a day."""
        feed = _parse([_good(normalized_sha256="0" * 64)])
        assert len(feed.test) == 1
        assert feed.hash_mismatches == 1

    def test_every_test_line_rejected_stops_the_day(self) -> None:
        """A feed without the marker is a format mismatch, not bad content."""
        with pytest.raises(FeedFileError, match="every one of 2 test lines"):
            _parse([_good("t1", verified=False), _good("t2", verified=False)])

    def test_garbage_lines_are_rejected_one_by_one(self) -> None:
        feed = _parse([_good("t0"), b"", b"[1, 2]", b"{not json"])
        assert sorted(rejected.reason for rejected in feed.rejected) == [
            "blank line",
            "not a JSON object",
            "not valid JSON",
        ]

    def test_extra_fields_on_a_test_line_are_ignored(self) -> None:
        feed = _parse([_good(group_slug="the-group")])
        assert len(feed.test) == 1


class TestProductionLines:
    def test_a_conforming_line_is_accepted(self) -> None:
        feed = _parse([_good()], [production_line("p1", "an ordinary post")])
        assert [line.item_id for line in feed.production] == ["p1"]

    def test_an_explicit_verified_marker_is_allowed(self) -> None:
        feed = _parse([_good()], [production_line("p1", "post", verified=True)])
        assert len(feed.production) == 1

    @pytest.mark.parametrize(
        "extra",
        [{"author": "anon"}, {"miner_hotkey": MINER_B.ss58_address}, {"group_slug": "g"}],
    )
    def test_any_attribution_rejects_the_line(self, extra: dict[str, Any]) -> None:
        """Production content is unattributed. A line that names anyone is refused."""
        feed = _parse([_good()], [production_line("p1", "post", **extra)])
        assert not feed.production
        assert "unattributed" in feed.rejected[0].reason

    def test_verified_false_rejects_the_line(self) -> None:
        feed = _parse([_good()], [production_line("p1", "post", verified=False)])
        assert not feed.production


class TestIds:
    def test_a_repeated_id_keeps_the_first_line(self) -> None:
        feed = _parse([_good("t1"), _good("t1")])
        assert len(feed.test) == 1
        assert feed.rejected[0].reason == "id repeats an earlier line"

    def test_ids_are_unique_across_both_files(self) -> None:
        feed = _parse([_good("x1")], [production_line("x1", "post")])
        assert not feed.production
        assert feed.rejected[0].file == "production"


class TestTimestamps:
    @pytest.mark.parametrize(
        ("text", "micros"),
        [
            ("1970-01-01T00:00:00Z", 0),
            ("1970-01-01T00:00:01.5Z", 1_500_000),
            ("1970-01-01T00:00:00.123456789Z", 123_456),
            ("1970-01-01T01:00:00+01:00", 0),
            ("1969-12-31T23:00:00-01:00", 0),
        ],
    )
    def test_rfc_3339_becomes_exact_integer_microseconds(self, text: str, micros: int) -> None:
        assert parse_timestamp_micros(text) == micros

    @pytest.mark.parametrize(
        "text", ["2026-11-05", "2026-11-05T12:00:00", "2026-13-05T12:00:00Z", 1700000000, None]
    )
    def test_anything_else_is_refused(self, text: Any) -> None:
        assert parse_timestamp_micros(text) is None
