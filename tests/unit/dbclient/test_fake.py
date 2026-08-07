"""The fake and the client, driven against each other over the real contract.

These are the tests that make the validator ingest path trustworthy before the
real DB layer exists: a real signature is produced by the client and checked by
the fake, and a real content hash is produced by the fake and recomputed by the
client. Nothing here is stubbed, so nothing here can pass by agreeing with
itself.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair
from tests.unit.dbclient.factories import (
    BASE_URL,
    CYCLE_AT,
    DAY,
    NETUID,
    Clock,
    make_eligible,
    make_production_items,
    make_test_items,
)

from prometheon.dbclient import (
    CallerRole,
    DbClient,
    FakeDbLayer,
    ModelStatus,
    auth,
)
from prometheon.dbclient.models import EvaluationSubmission, MinerResult
from prometheon.errors import (
    DbLayerError,
    EmbargoedError,
    NotAuthorizedError,
    SnapshotNotReadyError,
)

pytestmark = pytest.mark.unit

SECONDS_PER_DAY = 86_400


class TestIngest:
    def test_a_whole_day_round_trips_and_verifies_its_own_hash(
        self, seeded: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        snapshot = client.fetch_day(DAY)
        assert snapshot.manifest.date == DAY
        assert len(snapshot.test_items) == 4
        assert len(snapshot.production_items) == 7
        assert [miner.hotkey for miner in snapshot.eligible_miners] == [miner_keypair.ss58_address]
        assert len(snapshot.model_commitments) == 1

    def test_test_content_keeps_its_attribution(
        self, seeded: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        """Self-exclusion and the dataset reward both depend on this surviving."""
        items = client.fetch_test_content(DAY)
        assert {item.author_hotkey for item in items} == {miner_keypair.ss58_address}

    def test_a_server_that_attributes_production_content_is_refused(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        """Knowing whose traffic a negative came from would be a scoring lever."""

        def handler(request: httpx.Request) -> httpx.Response:
            response = seeded.handler(request)
            body = response.json()
            items = body.get("items") if response.status_code == 200 else None
            if not items or "observed_at" not in items[0]:
                return response
            leaked = [{**item, "author_hotkey": validator_keypair.ss58_address} for item in items]
            return httpx.Response(200, json={**body, "items": leaked})

        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=httpx.MockTransport(handler),
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_production_content(DAY)
        assert "cannot accept" in str(caught.value)

    def test_the_commitment_set_declares_its_freeze_instant(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        commitments = client.get_model_commitments(DAY)
        midnight = int(dt.datetime.combine(DAY, dt.time.min, tzinfo=dt.timezone.utc).timestamp())
        assert commitments.frozen_at == midnight

    def test_an_unbuilt_snapshot_is_reported_as_not_ready(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        seeded.mark_not_ready(DAY)
        with pytest.raises(SnapshotNotReadyError) as caught:
            client.get_snapshot(DAY)
        assert caught.value.status_code == 409

    def test_an_unbuilt_snapshot_also_withholds_the_dataset(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        seeded.mark_not_ready(DAY)
        with pytest.raises(SnapshotNotReadyError):
            client.fetch_test_content(DAY)

    def test_a_day_that_was_never_built_is_a_plain_not_found(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        with pytest.raises(DbLayerError) as caught:
            client.get_snapshot(DAY + dt.timedelta(days=5))
        assert caught.value.status_code == 404
        assert not isinstance(caught.value, SnapshotNotReadyError)


class TestPagination:
    def test_every_item_arrives_exactly_once_across_pages(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        items = make_test_items(23, author=miner_keypair.ss58_address)
        fake.max_page_limit = 5
        fake.seed_day(DAY, test_items=items)
        fetched = client.fetch_test_content(DAY)
        assert [item.id for item in fetched] == [item.id for item in items]

    def test_production_paginates_too(self, fake: FakeDbLayer, client: DbClient) -> None:
        fake.max_page_limit = 3
        fake.seed_day(DAY, production_items=make_production_items(10))
        assert len(client.fetch_production_content(DAY)) == 10

    def test_the_last_page_carries_no_cursor(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        fake.seed_day(DAY, test_items=make_test_items(2, author=miner_keypair.ss58_address))
        page = client.get_test_page(DAY)
        assert page.next_cursor is None
        assert page.total_count == 2

    def test_an_exactly_full_page_does_not_promise_another(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        """The off-by-one that would make a client fetch an empty final page."""
        fake.max_page_limit = 4
        fake.seed_day(DAY, test_items=make_test_items(4, author=miner_keypair.ss58_address))
        assert client.get_test_page(DAY).next_cursor is None
        assert len(fake.request_log) == 1

    def test_an_empty_collection_is_one_page(self, seeded: FakeDbLayer, client: DbClient) -> None:
        seeded.seed_day(DAY, test_items=())
        assert client.fetch_test_content(DAY) == ()

    def test_a_cursor_from_another_collection_is_refused(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        fake.max_page_limit = 2
        fake.seed_day(
            DAY,
            test_items=make_test_items(5, author=miner_keypair.ss58_address),
            production_items=make_production_items(5),
        )
        cursor = client.get_test_page(DAY).next_cursor
        assert cursor is not None
        with pytest.raises(DbLayerError) as caught:
            client.get_production_page(DAY, cursor=cursor)
        assert caught.value.error_code == "db.bad_cursor"

    def test_a_cursor_from_another_day_is_refused(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        other = DAY - dt.timedelta(days=1)
        fake.max_page_limit = 2
        fake.seed_day(DAY, test_items=make_test_items(5, author=miner_keypair.ss58_address))
        fake.seed_day(other, test_items=make_test_items(5, author=miner_keypair.ss58_address))
        cursor = client.get_test_page(DAY).next_cursor
        assert cursor is not None
        with pytest.raises(DbLayerError) as caught:
            client.get_test_page(other, cursor=cursor)
        assert caught.value.error_code == "db.bad_cursor"

    def test_a_limit_beyond_the_contract_ceiling_is_refused_by_the_server(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        """The client also guards this, so the check has to be made directly."""
        headers = auth.signed_headers(
            validator_keypair,
            method="GET",
            path="/v2/dataset/2026-08-05/test?limit=5000",
            body=b"",
            netuid=NETUID,
            hotkey=validator_keypair.ss58_address,
            nonce=auth.new_nonce(),
            issued_at=clock(),
        )
        response = seeded.handler(
            httpx.Request(
                "GET", f"{BASE_URL}/v2/dataset/2026-08-05/test?limit=5000", headers=headers
            )
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "db.bad_request"


class TestEmbargo:
    def test_a_miner_cannot_read_content_on_the_day_after(
        self, seeded: FakeDbLayer, miner_client: DbClient
    ) -> None:
        with pytest.raises(EmbargoedError) as caught:
            miner_client.fetch_test_content(DAY)
        assert caught.value.status_code == 403

    def test_the_embargo_covers_production_content_too(
        self, seeded: FakeDbLayer, miner_client: DbClient
    ) -> None:
        with pytest.raises(EmbargoedError):
            miner_client.fetch_production_content(DAY)

    def test_it_lifts_at_midnight_two_days_later(
        self, seeded: FakeDbLayer, miner_client: DbClient, clock: Clock
    ) -> None:
        lifts_at = int(
            dt.datetime.combine(
                DAY + dt.timedelta(days=2), dt.time.min, tzinfo=dt.timezone.utc
            ).timestamp()
        )
        clock.now = lifts_at - 1
        with pytest.raises(EmbargoedError):
            miner_client.fetch_test_content(DAY)
        clock.now = lifts_at
        assert len(miner_client.fetch_test_content(DAY)) == 4

    def test_a_validator_reads_the_same_day_immediately(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        assert client.fetch_test_content(DAY)

    def test_the_embargo_error_says_when_the_content_becomes_readable(
        self, seeded: FakeDbLayer, miner_client: DbClient
    ) -> None:
        lifts_at = int(
            dt.datetime.combine(
                DAY + dt.timedelta(days=2), dt.time.min, tzinfo=dt.timezone.utc
            ).timestamp()
        )
        with pytest.raises(EmbargoedError) as caught:
            miner_client.fetch_test_content(DAY)
        assert str(lifts_at) in str(caught.value)


class TestAuthorisation:
    def test_a_miner_may_not_read_the_manifest(
        self, seeded: FakeDbLayer, miner_client: DbClient
    ) -> None:
        with pytest.raises(NotAuthorizedError):
            miner_client.get_snapshot(DAY)

    def test_a_miner_may_not_read_the_eligible_miner_list(
        self, seeded: FakeDbLayer, miner_client: DbClient
    ) -> None:
        with pytest.raises(NotAuthorizedError):
            miner_client.get_eligible_miners(DAY)

    def test_an_unregistered_hotkey_reaches_nothing(
        self, seeded: FakeDbLayer, stranger_keypair: Keypair, clock: Clock
    ) -> None:
        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=stranger_keypair,
            transport=seeded.transport,
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(NotAuthorizedError) as caught:
                db.get_snapshot(DAY)
        assert caught.value.status_code == 403

    def test_a_replayed_nonce_is_rejected_end_to_end(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        """A client that pinned its nonce would look exactly like an attacker."""
        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=seeded.transport,
            now=clock,
            nonce_factory=lambda: "f" * 32,
            sleep=lambda _seconds: None,
        ) as db:
            db.get_snapshot(DAY)
            with pytest.raises(NotAuthorizedError) as caught:
                db.get_snapshot(DAY)
        assert caught.value.error_code == "db.auth_replayed"

    def test_a_client_on_the_wrong_netuid_is_rejected(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID + 1,
            keypair=validator_keypair,
            transport=seeded.transport,
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(NotAuthorizedError):
                db.get_snapshot(DAY)

    def test_a_skewed_client_clock_is_rejected(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=seeded.transport,
            now=Clock(CYCLE_AT + 400),
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(NotAuthorizedError) as caught:
                db.get_snapshot(DAY)
        assert caught.value.error_code == "db.auth_expired"


def _submission(hotkey: str, *, netuid: int = NETUID, burned: int = 0) -> EvaluationSubmission:
    return EvaluationSubmission(
        date=DAY,
        netuid=netuid,
        validator_hotkey=hotkey,
        scoring_version="prometheon-scoring/2.0.0",
        policy_version="2026-08-07",
        snapshot_content_hash="a" * 64,
        corpus_content_hash="b" * 64,
        labelled_test_count=4,
        labelled_production_count=7,
        results=(
            MinerResult(
                hotkey=hotkey,
                uid=0,
                weight=65_535,
                dataset_submitted_count=4,
                dataset_valid_count=3,
                dataset_score_micro=2_250_000,
                model_status=ModelStatus.EVALUATED,
                model_items_scored=11,
                model_correct_count=10,
                raw_accuracy_bp=9090,
                mean_total_tokens_milli=850_000,
                efficiency_penalty_bp=1200,
                model_score_bp=7999,
                model_rank=1,
            ),
        ),
        burned_weight=burned,
        started_at=CYCLE_AT,
        completed_at=CYCLE_AT + 900,
    )


class TestEvaluationSubmission:
    def test_a_signed_result_is_stored(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        receipt = client.submit_evaluation(signed)
        assert receipt.duplicate is False
        assert seeded.stored_evaluations() == (signed,)

    def test_resending_the_identical_result_is_a_success(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        """A client that lost the response needs a safe way to find out."""
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        first = client.submit_evaluation(signed)
        second = client.submit_evaluation(signed)
        assert second.duplicate is True
        assert second.accepted_at == first.accepted_at
        assert len(seeded.stored_evaluations()) == 1

    def test_revising_a_published_result_is_refused(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        hotkey = validator_keypair.ss58_address
        client.submit_evaluation(auth.sign_evaluation(validator_keypair, _submission(hotkey)))
        revised = auth.sign_evaluation(validator_keypair, _submission(hotkey, burned=5000))
        with pytest.raises(DbLayerError) as caught:
            client.submit_evaluation(revised)
        assert caught.value.status_code == 409
        assert caught.value.error_code == "db.duplicate_evaluation"

    def test_a_miner_may_not_publish_an_evaluation(
        self, seeded: FakeDbLayer, miner_client: DbClient, miner_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(miner_keypair, _submission(miner_keypair.ss58_address))
        with pytest.raises(NotAuthorizedError):
            miner_client.submit_evaluation(signed)

    def test_publishing_under_another_validators_name_is_refused(
        self, seeded: FakeDbLayer, client: DbClient, stranger_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(stranger_keypair, _submission(stranger_keypair.ss58_address))
        with pytest.raises(NotAuthorizedError) as caught:
            client.submit_evaluation(signed)
        assert caught.value.status_code == 403

    def test_a_tampered_result_is_refused(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        tampered = signed.model_copy(
            update={"results": (signed.results[0].model_copy(update={"weight": 1}),)}
        )
        with pytest.raises(DbLayerError) as caught:
            client.submit_evaluation(tampered)
        assert caught.value.status_code == 422
        assert seeded.stored_evaluations() == ()

    def test_a_result_for_another_netuid_is_refused(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address, netuid=NETUID + 1)
        )
        with pytest.raises(DbLayerError) as caught:
            client.submit_evaluation(signed)
        assert caught.value.status_code == 400


class TestContentHashAgreement:
    def _tampering_transport(self, fake: FakeDbLayer, mutate: Any) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            response = fake.handler(request)
            if response.status_code != 200:
                return response
            body = response.json()
            mutated = mutate(request, body)
            if mutated is None:
                return response
            return httpx.Response(200, json=mutated)

        return httpx.MockTransport(handler)

    def test_a_manifest_that_does_not_match_the_data_is_caught(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        """The one check that catches a DB layer serving two different corpora."""

        def mutate(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any] | None:
            if "content_hash" not in body:
                return None
            return {**body, "content_hash": "c" * 64}

        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=self._tampering_transport(seeded, mutate),
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_day(DAY)
        assert "content hash mismatch" in str(caught.value)

    def test_a_swapped_item_is_caught_even_when_the_counts_still_add_up(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        def mutate(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any] | None:
            items = body.get("items")
            if not items or "author_hotkey" not in items[0]:
                return None
            swapped = [{**items[0], "content": "quietly different"}, *items[1:]]
            return {**body, "items": swapped}

        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=self._tampering_transport(seeded, mutate),
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_day(DAY)
        assert "content hash mismatch" in str(caught.value)

    def test_a_short_count_is_caught_before_the_hash(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        def mutate(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any] | None:
            items = body.get("items")
            if not items or "author_hotkey" not in items[0]:
                return None
            return {**body, "items": items[1:]}

        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=self._tampering_transport(seeded, mutate),
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_day(DAY)
        assert "test_item_count" in str(caught.value)

    def test_verification_can_be_skipped_deliberately(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        def mutate(request: httpx.Request, body: dict[str, Any]) -> dict[str, Any] | None:
            if "content_hash" not in body:
                return None
            return {**body, "content_hash": "c" * 64}

        with DbClient(
            base_url=BASE_URL,
            netuid=NETUID,
            keypair=validator_keypair,
            transport=self._tampering_transport(seeded, mutate),
            now=clock,
            sleep=lambda _seconds: None,
        ) as db:
            assert db.fetch_day(DAY, verify_content_hash=False).manifest.content_hash == "c" * 64


class TestSeeding:
    def test_the_production_cap_is_enforced_at_seed_time(self, fake: FakeDbLayer) -> None:
        with pytest.raises(DbLayerError):
            fake.seed_day(DAY, production_items=make_production_items(10_001))

    def test_the_default_build_instant_is_the_following_midnight(
        self, fake: FakeDbLayer, client: DbClient, miner_keypair: Keypair
    ) -> None:
        """A day cannot be complete before it is over."""
        fake.seed_day(DAY, eligible_miners=(make_eligible(miner_keypair.ss58_address),))
        midnight = int(dt.datetime.combine(DAY, dt.time.min, tzinfo=dt.timezone.utc).timestamp())
        assert client.get_snapshot(DAY).built_at == midnight + SECONDS_PER_DAY

    def test_the_request_log_records_what_went_on_the_wire(
        self, seeded: FakeDbLayer, client: DbClient
    ) -> None:
        client.get_snapshot(DAY)
        client.get_eligible_miners(DAY)
        assert seeded.request_log == [
            ("GET", "/v2/snapshot/2026-08-05"),
            ("GET", "/v2/eligible-miners/2026-08-05"),
        ]

    def test_an_unknown_endpoint_is_a_404(
        self, seeded: FakeDbLayer, validator_keypair: Keypair, clock: Clock
    ) -> None:
        headers = auth.signed_headers(
            validator_keypair,
            method="GET",
            path="/v2/nope",
            body=b"",
            netuid=NETUID,
            hotkey=validator_keypair.ss58_address,
            nonce=auth.new_nonce(),
            issued_at=clock(),
        )
        response = seeded.handler(httpx.Request("GET", f"{BASE_URL}/v2/nope", headers=headers))
        assert response.status_code == 404

    def test_registration_can_be_revoked(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        assert client.get_snapshot(DAY)
        seeded.deregister(validator_keypair.ss58_address)
        with pytest.raises(NotAuthorizedError):
            client.get_snapshot(DAY)

    def test_losing_a_validator_permit_closes_the_manifest_immediately(
        self, seeded: FakeDbLayer, client: DbClient, validator_keypair: Keypair
    ) -> None:
        assert client.get_snapshot(DAY)
        seeded.register(validator_keypair.ss58_address, CallerRole.MINER)
        with pytest.raises(NotAuthorizedError):
            client.get_snapshot(DAY)
