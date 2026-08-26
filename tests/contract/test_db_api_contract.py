"""Pins the byte-level examples in ``docs/db-api-contract.md``.

The DB layer is built by another team against that document. If the values
below change, every conforming implementation stops agreeing with this one, and
the document is wrong until it is updated to match. These tests exist so that
happens loudly, in review, rather than quietly in production when two
validators derive different hashes from the same day.

Nothing here is a re-implementation of the algorithm. They are literals copied
out of the specification.
"""

from __future__ import annotations

import datetime as dt

import pytest

from prometheon.dbclient import auth, models
from prometheon.security.canonical_json import DOMAIN_DB_REQUEST, domain_prefixed_bytes

pytestmark = pytest.mark.contract

HOTKEY = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
DAY = dt.date(2026, 8, 5)

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

SIGNED_BYTES = (
    b"PROMETHEON_V2_DB_REQUEST\n"
    b'{"body_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",'
    b'"domain":"PROMETHEON_V2_DB_REQUEST",'
    b'"hotkey":"5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",'
    b'"issued_at":1786032000,'
    b'"method":"GET",'
    b'"netuid":42,'
    b'"nonce":"9f1c0d3a7b5e42610fd8c2b41a6e9037",'
    b'"path":"/v2/dataset/2026-08-05/test'
    b'?cursor=eyJjIjoidGVzdCIsImQiOiIyMDI2LTA4LTA1IiwibyI6NTAwfQ&limit=500"}'
)

TEST_ITEM_DIGEST = "dc3cb7312c042b4c037f7a4c69c9c447c92c339ca4c00c9b673023e91b989aa0"
TEST_COLLECTION_HASH = "6217b30bab8e8e6049f9d0dc1105422fc5bea4179794bcb70bcaab73a0739491"
WORKED_CONTENT_HASH = "6ea0849e4fef4f8d2881180e63b9a95013b94a7bb6b24f55a381878f55aa3b53"
EMPTY_DAY_CONTENT_HASH = "f1afe099299e07dabf31b211b6dcee506d5d5fe95333978154c74c5becb77f92"


def _test_item() -> models.TestContentItem:
    return models.TestContentItem(
        id="t1",
        content="example",
        author_hotkey=HOTKEY,
        claimed_violating=True,
        submitted_at=1_786_000_000,
    )


def _production_item() -> models.ProductionContentItem:
    return models.ProductionContentItem(id="p1", content="hello", observed_at=1_786_000_001)


def _eligible() -> models.EligibleMiner:
    return models.EligibleMiner(hotkey=HOTKEY, qualified_user_count=12)


def _commitment() -> models.ModelCommitment:
    return models.ModelCommitment(
        hotkey=HOTKEY,
        hf_repo="miner/model",
        revision_sha="a" * 40,
        committed_at=1_785_999_000,
        block=4242,
    )


class TestAuthEnvelope:
    def test_an_empty_body_hashes_to_the_documented_digest(self) -> None:
        assert auth.body_sha256(b"") == EMPTY_SHA256

    def test_the_signed_bytes_match_the_specification(self) -> None:
        envelope = auth.build_envelope(
            method="GET",
            path=(
                "/v2/dataset/2026-08-05/test"
                "?cursor=eyJjIjoidGVzdCIsImQiOiIyMDI2LTA4LTA1IiwibyI6NTAwfQ&limit=500"
            ),
            body=b"",
            netuid=42,
            hotkey=HOTKEY,
            nonce="9f1c0d3a7b5e42610fd8c2b41a6e9037",
            issued_at=1_786_032_000,
        )
        assert domain_prefixed_bytes(DOMAIN_DB_REQUEST, envelope.signing_payload()) == SIGNED_BYTES

    def test_the_documented_headers_are_the_ones_sent(self) -> None:
        assert set(auth.AUTH_HEADERS) == {
            "x-prometheon-netuid",
            "x-prometheon-hotkey",
            "x-prometheon-nonce",
            "x-prometheon-issued-at",
            "x-prometheon-body-sha256",
            "x-prometheon-signature",
        }

    def test_the_clock_window_is_five_minutes(self) -> None:
        assert auth.AUTH_MAX_CLOCK_SKEW_SECONDS == 300

    def test_nonce_retention_outlasts_two_skew_windows(self) -> None:
        assert auth.NONCE_RETENTION_SECONDS > 2 * auth.AUTH_MAX_CLOCK_SKEW_SECONDS


class TestSnapshotContentHash:
    def test_a_test_item_digest_matches_the_specification(self) -> None:
        assert models.test_item_digest(_test_item()).hex() == TEST_ITEM_DIGEST

    def test_a_single_member_collection_hash_matches(self) -> None:
        assert (
            models.collection_hash([models.test_item_digest(_test_item())]) == TEST_COLLECTION_HASH
        )

    def test_an_empty_collection_hashes_to_the_empty_string_digest(self) -> None:
        assert models.collection_hash([]) == EMPTY_SHA256

    def test_the_worked_example_matches(self) -> None:
        assert (
            models.snapshot_content_hash(
                date=DAY,
                test_items=[_test_item()],
                production_items=[_production_item()],
                eligible_miners=[_eligible()],
                model_commitments=[_commitment()],
            )
            == WORKED_CONTENT_HASH
        )

    def test_an_entirely_empty_day_matches(self) -> None:
        assert (
            models.snapshot_content_hash(
                date=DAY,
                test_items=[],
                production_items=[],
                eligible_miners=[],
                model_commitments=[],
            )
            == EMPTY_DAY_CONTENT_HASH
        )


class TestDocumentedConstants:
    def test_the_snapshot_version_is_the_one_in_the_hash_preimage(self) -> None:
        assert models.SNAPSHOT_VERSION == "prometheon-snapshot/2"

    def test_the_documented_limits_hold(self) -> None:
        assert models.PRODUCTION_ITEM_CAP == 10_000
        assert models.MAX_PAGE_LIMIT == 1_000
        assert models.DEFAULT_PAGE_LIMIT == 500
        assert models.MAX_CONTENT_CHARS == 32_000
        assert models.EMBARGO_DAYS == 2

    def test_every_documented_error_code_exists(self) -> None:
        assert {code.value for code in models.DbErrorCode} == {
            "db.bad_request",
            "db.bad_cursor",
            "db.auth_malformed",
            "db.auth_invalid",
            "db.auth_expired",
            "db.auth_replayed",
            "db.not_authorized",
            "db.embargoed",
            "db.not_found",
            "db.snapshot_not_ready",
            "db.duplicate_evaluation",
            "db.invalid_payload",
            "db.rate_limited",
            "db.internal",
        }
