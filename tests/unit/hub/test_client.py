"""The hub client against the fake hub, over real request and response objects."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from tests.unit.hub.factories import (
    DAY,
    MINER_A,
    NETUID,
    VALIDATOR,
    Clock,
    feed_test_line,
    standard_miners,
)

from prometheon.errors import (
    EpochMissingError,
    EpochNotReadyError,
    EpochVerificationError,
    FeedFileError,
    HubAuthError,
    HubError,
    RecordMissingError,
)
from prometheon.hub.auth import sign_evaluation, verify_evaluation
from prometheon.hub.client import HubClient
from prometheon.hub.fake import FakeHub, gzip_lines
from prometheon.hub.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.unit

ClientFactory = Callable[..., HubClient]


def _record(epoch_hash: str = "e" * 64, weight: int = 600_000) -> EvaluationSubmission:
    row = MinerResult(
        hotkey=MINER_A.ss58_address,
        uid=1,
        weight=weight,
        dataset_submitted_count=10,
        dataset_valid_count=10,
        dataset_violating_confirmed=5,
        dataset_compliant_confirmed=5,
        dataset_score_micro=10_000_000,
        model_status=ModelStatus.EVALUATED,
        model_items_scored=100,
        model_correct_count=90,
        model_parameters=494_032_768,
        raw_accuracy_bp=9000,
        size_penalty_bp=0,
        model_score_bp=9000,
        model_rank=1,
    )
    return sign_evaluation(
        VALIDATOR,
        EvaluationSubmission(
            date=DAY,
            netuid=NETUID,
            validator_hotkey=VALIDATOR.ss58_address,
            scoring_version=SCORING_VERSION,
            policy_version="2026-08-07",
            epoch_content_hash=epoch_hash,
            corpus_content_hash="c" * 64,
            labelled_test_count=2,
            labelled_production_count=1,
            results=(row,),
            burned_weight=1_000_000 - weight,
            started_at=1,
            completed_at=2,
        ),
    )


class TestFetchingADay:
    def test_a_whole_day_round_trips(self, make_client: ClientFactory) -> None:
        with make_client() as client:
            day = client.fetch_day(DAY, scoring_version=SCORING_VERSION)

        assert day.epoch.day == DAY
        assert [line.item_id for line in day.feed.test] == ["t1", "t2"]
        assert [line.item_id for line in day.feed.production] == ["p1"]
        assert len(day.verified.content_hash) == 64

    def test_the_request_is_signed_over_the_full_path_including_a_prefix(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        """The fake verifies the envelope, so a wrong signed path is a 401 here."""
        with make_client(base_url="https://hub.invalid/prefix") as client:
            client.request_epoch(DAY)
        assert ("POST", "/prefix/api/v1/validator/epoch") in hub.request_log

    def test_a_day_not_built_yet_says_so(self, make_client: ClientFactory, hub: FakeHub) -> None:
        hub.not_ready.add(DAY)
        with make_client() as client, pytest.raises(EpochNotReadyError):
            client.request_epoch(DAY)

    def test_a_day_that_will_never_exist_says_so(self, make_client: ClientFactory) -> None:
        with make_client() as client, pytest.raises(EpochMissingError):
            client.request_epoch(DAY - dt.timedelta(days=30))

    def test_a_miner_is_refused(self, make_client: ClientFactory) -> None:
        with make_client(keypair=MINER_A) as client, pytest.raises(HubAuthError):
            client.request_epoch(DAY)

    def test_no_hotkey_cannot_ask_for_an_epoch(self, make_client: ClientFactory) -> None:
        with make_client(keypair=None) as client, pytest.raises(HubError, match="needs a hotkey"):
            client.request_epoch(DAY)

    def test_an_unpinned_key_fails_before_anything_is_downloaded(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        with (
            make_client(epoch_keys={"someone-else": "00" * 32}) as client,
            pytest.raises(EpochVerificationError, match="does not pin"),
        ):
            client.fetch_day(DAY, scoring_version=SCORING_VERSION)
        assert not any("storage.invalid" in target for _, target in hub.request_log)

    def test_a_file_swapped_after_signing_stops_the_day(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        hub.replace_file(DAY, "test", gzip_lines([feed_test_line("t9", MINER_A.ss58_address, "x")]))
        with make_client() as client, pytest.raises(FeedFileError, match="hashes to"):
            client.fetch_day(DAY, scoring_version=SCORING_VERSION)

    def test_an_expired_url_fails_rather_than_serving_stale_bytes(
        self, make_client: ClientFactory, clock: Clock
    ) -> None:
        with make_client() as client:
            bundle = client.request_epoch(DAY)
            clock.now += 3601
            with pytest.raises(HubError, match="HTTP 403"):
                client.download(bundle.urls.test)

    def test_an_epoch_for_other_rules_is_refused(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        hub.publish_day(DAY, miners=standard_miners(), scoring_version="prometheon-scoring/9.9")
        with make_client() as client, pytest.raises(EpochVerificationError, match="frozen under"):
            client.fetch_day(DAY, scoring_version=SCORING_VERSION)

    def test_plain_http_downloads_are_refused(self, make_client: ClientFactory) -> None:
        with make_client() as client, pytest.raises(HubError, match="refusing"):
            client.download("http://storage.example/test.jsonl.gz")


class TestRetries:
    def _flaky(self, hub: FakeHub, failures: list[int]) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            if failures and request.url.host != "storage.invalid":
                return httpx.Response(failures.pop(0))
            return hub.handler(request)

        return httpx.MockTransport(handler)

    def test_a_shed_request_is_retried_with_a_fresh_nonce(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        """Re-sending a used envelope is a replay the hub must refuse."""
        used: list[str] = []

        def nonce() -> str:
            used.append(f"{len(used):032x}")
            return used[-1]

        with make_client(transport=self._flaky(hub, [503, 429]), nonce_factory=nonce) as client:
            client.request_epoch(DAY)
        assert len(used) == 3
        assert len(set(used)) == 3

    def test_a_transport_error_is_retried(self, make_client: ClientFactory, hub: FakeHub) -> None:
        calls = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.ConnectError("reset", request=request)
            return hub.handler(request)

        with make_client(transport=httpx.MockTransport(handler)) as client:
            client.request_epoch(DAY)
        assert calls["count"] == 2

    def test_retries_are_bounded(self, make_client: ClientFactory, hub: FakeHub) -> None:
        with (
            make_client(transport=self._flaky(hub, [503] * 10), max_retries=2) as client,
            pytest.raises(HubError, match="HTTP 503"),
        ):
            client.request_epoch(DAY)


class TestResults:
    def test_a_record_is_stored_once_and_a_resend_is_a_duplicate(
        self, make_client: ClientFactory
    ) -> None:
        record = _record()
        with make_client() as client:
            first = client.submit_result(record)
            again = client.submit_result(record)
        assert first.duplicate is False
        assert again.duplicate is True

    def test_a_different_record_for_the_same_day_is_refused(
        self, make_client: ClientFactory
    ) -> None:
        with make_client() as client:
            client.submit_result(_record(weight=600_000))
            with pytest.raises(HubError, match="409"):
                client.submit_result(_record(weight=500_000))

    def test_a_published_record_reads_back_verbatim_and_still_verifies(
        self, make_client: ClientFactory
    ) -> None:
        record = _record()
        with make_client() as client:
            client.submit_result(record)
        with make_client(keypair=None) as reader:
            fetched = reader.get_result(DAY, VALIDATOR.ss58_address)
        assert fetched == record
        verify_evaluation(fetched)

    def test_the_signed_payload_is_the_record_minus_its_signature(self) -> None:
        """What the hub verifies: every top-level field received except `signature`."""
        record = _record()
        on_the_wire = json.loads(record.model_dump_json())
        del on_the_wire["signature"]
        assert on_the_wire == record.signing_payload()

    def test_an_unsigned_record_is_not_sent(self, make_client: ClientFactory) -> None:
        unsigned = _record().model_copy(update={"signature": ""})
        with make_client() as client, pytest.raises(HubError, match="unsigned"):
            client.submit_result(unsigned)


class TestTheOwnerValidatorsWeights:
    """Copy mode's route: the owner validator's record, for validators only."""

    def _post(self, hub: FakeHub, make_client: ClientFactory) -> EvaluationSubmission:
        record = _record()
        hub.owner_validator = VALIDATOR.ss58_address
        with make_client() as client:
            client.submit_result(record)
        return record

    def test_the_record_comes_back_exactly_as_posted_and_still_verifies(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        posted = self._post(hub, make_client)
        with make_client() as client:
            fetched = client.request_weights(DAY)

        assert fetched == posted
        verify_evaluation(fetched)

    def test_the_request_is_signed_by_the_asking_validator(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        self._post(hub, make_client)
        with make_client() as client:
            client.request_weights(DAY)
        assert ("POST", "/api/v1/validator/weights") in hub.request_log

    def test_a_day_the_owner_has_not_posted_yet_is_not_ready(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        hub.owner_validator = VALIDATOR.ss58_address
        with make_client() as client, pytest.raises(EpochNotReadyError, match="retry after 600s"):
            client.request_weights(DAY)

    def test_a_day_that_will_never_have_a_record_says_so(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        hub.owner_validator = VALIDATOR.ss58_address
        with make_client() as client, pytest.raises(RecordMissingError):
            client.request_weights(DAY - dt.timedelta(days=30))

    def test_a_miner_cannot_ask(self, make_client: ClientFactory, hub: FakeHub) -> None:
        self._post(hub, make_client)
        with make_client(keypair=MINER_A) as client, pytest.raises(HubAuthError):
            client.request_weights(DAY)

    def test_nobody_unsigned_can_ask(self, make_client: ClientFactory, hub: FakeHub) -> None:
        self._post(hub, make_client)
        with make_client(keypair=None) as client, pytest.raises(HubError, match="needs a hotkey"):
            client.request_weights(DAY)


class TestPublicReads:
    def test_the_public_epoch_verifies_against_the_pinned_key(
        self, make_client: ClientFactory
    ) -> None:
        with make_client(keypair=None) as client:
            verified = client.get_public_epoch(DAY)
        assert verified.epoch.day == DAY

    def test_a_coldkey_with_no_ledger_has_a_zero_balance(self, make_client: ClientFactory) -> None:
        with make_client(keypair=None) as client:
            balance = client.get_credits(MINER_A.ss58_address)
        assert balance.balance == 0
        assert balance.entries == ()

    def test_the_ledger_is_read_as_published(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        hub.credits[MINER_A.ss58_address] = {
            "coldkey": MINER_A.ss58_address,
            "balance": 1,
            "remainder_alpha_rao": 10_000_000_000,
            "entries": [
                {
                    "kind": "deposit",
                    "block": 5,
                    "extrinsic_hash": "0xab",
                    "alpha_rao": 60_000_000_000,
                    "credits": 1,
                },
            ],
            "something_new": True,
        }
        with make_client(keypair=None) as client:
            balance = client.get_credits(MINER_A.ss58_address)
        assert balance.balance == 1
        assert balance.entries[0].extrinsic_hash == "0xab"

    def test_a_malformed_address_never_reaches_the_wire(
        self, make_client: ClientFactory, hub: FakeHub
    ) -> None:
        with make_client(keypair=None) as client, pytest.raises(HubError, match="SS58"):
            client.get_credits("../../etc/passwd")
        assert not hub.request_log


class TestTheFakeEnforcesTheEmbargo:
    """The public corpus: test content only from N+2, paged, with no author."""

    def _get(self, hub: FakeHub, query: str) -> httpx.Response:
        return hub.handler(httpx.Request("GET", f"https://hub.invalid/api/v1/corpus?{query}"))

    def test_before_n_plus_2_it_is_embargoed(self, hub: FakeHub) -> None:
        response = self._get(hub, f"day={DAY.isoformat()}")
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "db.embargoed"

    def test_from_n_plus_2_it_is_public_without_attribution(
        self, hub: FakeHub, clock: Clock
    ) -> None:
        clock.now = int(dt.datetime(2026, 11, 7, tzinfo=dt.timezone.utc).timestamp())
        response = self._get(hub, f"day={DAY.isoformat()}")
        items: list[dict[str, Any]] = response.json()["items"]
        assert [item["id"] for item in items] == ["t1", "t2"]
        assert all(set(item) == {"id", "body", "claimed_violating", "created_at"} for item in items)
