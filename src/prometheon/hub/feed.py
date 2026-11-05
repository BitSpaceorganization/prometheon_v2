"""Reading the day's content files, line by line, rejecting what the spec forbids.

The two files are BitFan's ``test.jsonl.gz`` and ``production.jsonl.gz``, passed
through the hub unchanged. The epoch carries each file's sha256 and line count,
and both are checked before a line is read: a file that does not match is not a
file this day was frozen with, and the cycle stops.

After that, every line is checked on its own and a bad one is **rejected**:
dropped for this day, on every validator alike, and counted for the operator
without echoing its content. Rejecting a line rather than the whole file keeps
one malformed record from costing every miner a day. But a non-empty test file
in which every line is rejected stops the cycle, because that is a format
mismatch, not a bad record, and scoring it would burn the whole data pool
quietly.

**Whose content counts.** Only identity-verified people's. BitFan's export
already filters on it; on top of that, a test line must carry ``verified: true``
and a per-day ``author`` pseudonym, and must name a hotkey in the epoch's miner
list. A validator cannot see anyone's identity, so the marker is BitFan's
statement, carried in a file the hub signs. That is the whole of the guarantee,
and it is stated in the docs in those words.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from prometheon.errors import FeedFileError
from prometheon.hub.models import (
    MAX_CONTENT_CHARS,
    MAX_ITEM_ID_CHARS,
    EpochFile,
    EpochMiner,
    is_hex64,
    is_ss58,
)
from prometheon.scoring.dedupe import normalized_sha256

#: Ceiling on one decompressed file. A day is a few hundred thousand short posts
#: at most; the cap only stops a corrupt or hostile archive from exhausting
#: memory, and the archive's hash has already been checked against the signed
#: epoch before it is decompressed.
MAX_DECOMPRESSED_BYTES: Final[int] = 1 << 30

#: Longest per-day author pseudonym accepted.
_MAX_AUTHOR_CHARS: Final[int] = 128

_TIMESTAMP_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z|[+-]\d{2}:\d{2})$"
)

_UNIX_EPOCH: Final[dt.datetime] = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

#: Production content is unattributed. These are the only keys a production
#: line may carry, so a line that names an author or a hotkey is rejected rather
#: than letting attribution reach the corpus.
_PRODUCTION_KEYS: Final[frozenset[str]] = frozenset({"id", "body", "created_at", "verified"})


@dataclass(frozen=True, slots=True)
class TestLine:
    """One verified person's test post."""

    __test__ = False  # not a pytest test class, despite the name

    line_number: int
    item_id: str
    miner_hotkey: str
    author: str
    claimed_violating: bool
    body: str
    created_at_micros: int


@dataclass(frozen=True, slots=True)
class ProductionLine:
    """One public post by a verified person, unattributed."""

    line_number: int
    item_id: str
    body: str
    created_at_micros: int


@dataclass(frozen=True, slots=True)
class RejectedLine:
    """Where a rejected line was, and why. Never its content."""

    file: str
    line_number: int
    reason: str


@dataclass(frozen=True)
class DayFeed:
    """The accepted lines of one day, and an account of what was not accepted."""

    test: tuple[TestLine, ...]
    production: tuple[ProductionLine, ...]
    rejected: tuple[RejectedLine, ...]
    #: Test lines whose ``normalized_sha256`` disagreed with this validator's own
    #: recomputation. Not rejected: the validator uses its own value. Counted so
    #: a normaliser drift between BitFan and the subnet is visible in the log.
    hash_mismatches: int = 0


def read_records(data: bytes, *, name: str, expected: EpochFile) -> list[bytes]:
    """Check a downloaded file against the epoch, then split it into records.

    Raises :class:`FeedFileError` when the hash or the line count does not
    match: that is not a bad line, it is not the file the day was frozen with.
    """
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected.sha256:
        raise FeedFileError(
            f"{name}.jsonl.gz hashes to {digest}, but the signed epoch says {expected.sha256}"
        )
    text = _gunzip(data, name=name)
    records = text.split(b"\n")
    if records and records[-1] == b"":
        records.pop()
    if len(records) != expected.count:
        raise FeedFileError(
            f"{name}.jsonl.gz holds {len(records)} lines, but the signed epoch says "
            f"{expected.count}"
        )
    return records


def parse_feed(
    *,
    test_records: Sequence[bytes],
    production_records: Sequence[bytes],
    day: dt.date,
    miners: Mapping[str, EpochMiner],
) -> DayFeed:
    """Every line checked on its own. Ids must be unique across both files."""
    rejected: list[RejectedLine] = []
    seen_ids: set[str] = set()
    tests: list[TestLine] = []
    mismatches = 0

    for number, record in enumerate(test_records, start=1):
        outcome = _test_line(record, number, day=day, miners=miners)
        if isinstance(outcome, str):
            rejected.append(RejectedLine("test", number, outcome))
            continue
        line, mismatch = outcome
        if line.item_id in seen_ids:
            rejected.append(RejectedLine("test", number, "id repeats an earlier line"))
            continue
        seen_ids.add(line.item_id)
        tests.append(line)
        mismatches += mismatch

    productions: list[ProductionLine] = []
    for number, record in enumerate(production_records, start=1):
        result = _production_line(record, number, day=day)
        if isinstance(result, str):
            rejected.append(RejectedLine("production", number, result))
            continue
        if result.item_id in seen_ids:
            rejected.append(RejectedLine("production", number, "id repeats an earlier line"))
            continue
        seen_ids.add(result.item_id)
        productions.append(result)

    if test_records and not tests:
        reasons = sorted({line.reason for line in rejected if line.file == "test"})
        raise FeedFileError(
            f"every one of {len(test_records)} test lines was rejected ({'; '.join(reasons)}). "
            "That is a format mismatch with the feed, not bad content, and scoring it would "
            "burn the whole data pool. Refusing the day"
        )

    return DayFeed(
        test=tuple(tests),
        production=tuple(productions),
        rejected=tuple(rejected),
        hash_mismatches=mismatches,
    )


def parse_timestamp_micros(value: object) -> int | None:
    """RFC 3339 to integer microseconds since the epoch, or ``None``.

    Integer arithmetic throughout: the result orders near-duplicates, and a
    float could put two items in a different order on two machines. Fractions
    beyond microseconds are truncated.
    """
    if not isinstance(value, str):
        return None
    match = _TIMESTAMP_RE.match(value)
    if match is None:
        return None
    zone = match.group("zone")
    try:
        offset = (
            dt.timezone.utc
            if zone == "Z"
            else dt.timezone(
                (1 if zone[0] == "+" else -1)
                * dt.timedelta(hours=int(zone[1:3]), minutes=int(zone[4:6]))
            )
        )
        moment = dt.datetime.combine(
            dt.date.fromisoformat(match.group("date")),
            dt.time(
                int(match.group("hour")),
                int(match.group("minute")),
                int(match.group("second")),
                int((match.group("fraction") or "0").ljust(6, "0")[:6]),
            ),
            tzinfo=offset,
        )
    except ValueError:
        return None
    return (moment - _UNIX_EPOCH) // dt.timedelta(microseconds=1)


def _day_bounds_micros(day: dt.date) -> tuple[int, int]:
    start = dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc)
    micro = dt.timedelta(microseconds=1)
    return (start - _UNIX_EPOCH) // micro, (start + dt.timedelta(days=1) - _UNIX_EPOCH) // micro


def _gunzip(data: bytes, *, name: str) -> bytes:
    """Decompress every gzip member, stopping at the ceiling."""
    out = bytearray()
    remaining = data
    while remaining:
        inflater = zlib.decompressobj(wbits=31)
        try:
            chunk = inflater.decompress(remaining, MAX_DECOMPRESSED_BYTES - len(out) + 1)
        except zlib.error as exc:
            raise FeedFileError(f"{name}.jsonl.gz is not valid gzip: {exc}") from exc
        out += chunk
        if len(out) > MAX_DECOMPRESSED_BYTES or inflater.unconsumed_tail:
            raise FeedFileError(
                f"{name}.jsonl.gz decompresses past the {MAX_DECOMPRESSED_BYTES} byte ceiling"
            )
        if not inflater.eof:
            raise FeedFileError(f"{name}.jsonl.gz is truncated")
        remaining = inflater.unused_data
    return bytes(out)


def _object(record: bytes) -> dict[str, Any] | str:
    if not record.strip():
        return "blank line"
    try:
        value = json.loads(record)
    except (ValueError, UnicodeDecodeError):
        return "not valid JSON"
    if not isinstance(value, dict):
        return "not a JSON object"
    return value


def _text(value: object, *, limit: int) -> str | None:
    """A non-empty string that is valid Unicode and within ``limit`` characters."""
    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        # A lone surrogate survives JSON parsing and breaks hashing later.
        return None
    return value


def _common(raw: Mapping[str, Any], *, day: dt.date) -> tuple[str, str, int] | str:
    item_id = _text(raw.get("id"), limit=MAX_ITEM_ID_CHARS)
    if item_id is None:
        return f"id missing, empty or over {MAX_ITEM_ID_CHARS} characters"
    body = _text(raw.get("body"), limit=MAX_CONTENT_CHARS)
    if body is None:
        return f"body missing, empty, over {MAX_CONTENT_CHARS} characters or not valid text"
    created = parse_timestamp_micros(raw.get("created_at"))
    if created is None:
        return "created_at is not an RFC 3339 timestamp"
    start, end = _day_bounds_micros(day)
    if not start <= created < end:
        return f"created_at falls outside {day.isoformat()} UTC"
    return item_id, body, created


def _test_line(
    record: bytes, number: int, *, day: dt.date, miners: Mapping[str, EpochMiner]
) -> tuple[TestLine, int] | str:
    raw = _object(record)
    if isinstance(raw, str):
        return raw
    if raw.get("verified") is not True:
        return "no verified marker"
    common = _common(raw, day=day)
    if isinstance(common, str):
        return common
    item_id, body, created = common
    hotkey = raw.get("miner_hotkey")
    if not is_ss58(hotkey):
        return "miner_hotkey is not an SS58 address"
    assert isinstance(hotkey, str)
    if hotkey not in miners:
        return "miner_hotkey is not in the epoch's miner list"
    author = _text(raw.get("author"), limit=_MAX_AUTHOR_CHARS)
    if author is None:
        return "author pseudonym missing"
    claimed = raw.get("claimed_violating")
    if not isinstance(claimed, bool):
        return "claimed_violating is not a boolean"
    declared = raw.get("normalized_sha256")
    if not is_hex64(declared):
        return "normalized_sha256 missing or not 64 lowercase hex"
    line = TestLine(
        line_number=number,
        item_id=item_id,
        miner_hotkey=hotkey,
        author=author,
        claimed_violating=claimed,
        body=body,
        created_at_micros=created,
    )
    return line, int(declared != normalized_sha256(body))


def _production_line(record: bytes, number: int, *, day: dt.date) -> ProductionLine | str:
    raw = _object(record)
    if isinstance(raw, str):
        return raw
    extra = sorted(set(raw) - _PRODUCTION_KEYS)
    if extra:
        return f"production lines are unattributed, but this one carries {extra}"
    if "verified" in raw and raw["verified"] is not True:
        return "verified is present but not true"
    common = _common(raw, day=day)
    if isinstance(common, str):
        return common
    item_id, body, created = common
    return ProductionLine(line_number=number, item_id=item_id, body=body, created_at_micros=created)


__all__ = [
    "MAX_DECOMPRESSED_BYTES",
    "DayFeed",
    "ProductionLine",
    "RejectedLine",
    "TestLine",
    "parse_feed",
    "parse_timestamp_micros",
    "read_records",
]
