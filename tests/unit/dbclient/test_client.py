"""Behaviour of the HTTP client: retries, error mapping, and path signing."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
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
    RecordingSleep,
)

from prometheon.config import ChainConfig, ChainNetwork, Config, DbLayerConfig, WalletConfig
from prometheon.dbclient import auth
from prometheon.dbclient.client import RETRYABLE_STATUSES, DbClient
from prometheon.dbclient.models import (
    SNAPSHOT_VERSION,
    DbErrorCode,
    EvaluationSubmission,
    MinerResult,
    ModelStatus,
    snapshot_content_hash,
)
from prometheon.errors import (
    DbLayerError,
    EmbargoedError,
    NotAuthorizedError,
    SnapshotNotReadyError,
)

pytestmark = pytest.mark.unit

Script = list[tuple[int, Any]]


def _manifest_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "date": DAY.isoformat(),
        "snapshot_version": SNAPSHOT_VERSION,
        "built_at": CYCLE_AT,
        "test_item_count": 0,
        "production_item_count": 0,
        "eligible_miner_count": 0,
        "model_commitment_count": 0,
        "content_hash": "0" * 64,
    }
    body.update(overrides)
    return body


def _error_body(code: str, **detail: Any) -> dict[str, Any]:
    return {"error": {"code": code, "message": "nope", **detail}}


def _scripted(script: Script) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """A transport that walks a script, repeating its last step forever."""
    seen: list[httpx.Request] = []
    remaining = list(script)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        status, body = remaining.pop(0) if len(remaining) > 1 else remaining[0]
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), seen


def _client(
    transport: httpx.BaseTransport,
    keypair: Keypair,
    *,
    base_url: str = BASE_URL,
    max_retries: int = 3,
    sleep: Callable[[float], None] | None = None,
) -> DbClient:
    return DbClient(
        base_url=base_url,
        netuid=NETUID,
        keypair=keypair,
        transport=transport,
        now=Clock(CYCLE_AT),
        max_retries=max_retries,
        sleep=sleep or (lambda _seconds: None),
    )


class TestRetries:
    @pytest.mark.parametrize("status", sorted(RETRYABLE_STATUSES))
    def test_a_retryable_status_is_retried(self, status: int, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(status, {}), (200, _manifest_body())])
        with _client(transport, validator_keypair) as db:
            assert db.get_snapshot(DAY).content_hash == "0" * 64
        assert len(seen) == 2

    def test_every_attempt_carries_a_fresh_nonce(self, validator_keypair: Keypair) -> None:
        """A reused envelope is a replay, and the server is required to reject it."""
        transport, seen = _scripted([(503, {}), (503, {}), (200, _manifest_body())])
        with _client(transport, validator_keypair) as db:
            db.get_snapshot(DAY)
        nonces = [request.headers[auth.HEADER_NONCE] for request in seen]
        assert len(set(nonces)) == len(seen) == 3

    def test_every_attempt_is_signed_afresh(self, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(503, {}), (200, _manifest_body())])
        with _client(transport, validator_keypair) as db:
            db.get_snapshot(DAY)
        signatures = {request.headers[auth.HEADER_SIGNATURE] for request in seen}
        assert len(signatures) == 2

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422, 500])
    def test_other_statuses_are_not_retried(self, status: int, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(status, _error_body("db.internal"))])
        with _client(transport, validator_keypair) as db, pytest.raises(DbLayerError):
            db.get_snapshot(DAY)
        assert len(seen) == 1

    def test_retries_are_bounded(self, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(503, _error_body("db.internal"))])
        with _client(transport, validator_keypair, max_retries=2) as db:
            with pytest.raises(DbLayerError) as caught:
                db.get_snapshot(DAY)
        assert len(seen) == 3
        assert caught.value.status_code == 503

    def test_max_retries_zero_sends_one_request(self, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(503, {})])
        with _client(transport, validator_keypair, max_retries=0) as db:
            with pytest.raises(DbLayerError):
                db.get_snapshot(DAY)
        assert len(seen) == 1

    def test_backoff_grows_between_attempts(self, validator_keypair: Keypair) -> None:
        sleep = RecordingSleep()
        transport, _seen = _scripted([(503, {}), (503, {}), (200, _manifest_body())])
        with _client(transport, validator_keypair, sleep=sleep) as db:
            db.get_snapshot(DAY)
        assert sleep.calls == sorted(sleep.calls)
        assert sleep.calls[0] < sleep.calls[-1]

    def test_a_numeric_retry_after_is_honoured(self, validator_keypair: Keypair) -> None:
        sleep = RecordingSleep()
        seen: list[httpx.Request] = []
        served = {"count": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            served["count"] += 1
            if served["count"] == 1:
                return httpx.Response(429, json={}, headers={"retry-after": "7"})
            return httpx.Response(200, json=_manifest_body())

        with _client(httpx.MockTransport(handler), validator_keypair, sleep=sleep) as db:
            db.get_snapshot(DAY)
        assert sleep.calls == [7.0]

    def test_an_absurd_retry_after_falls_back_to_backoff(self, validator_keypair: Keypair) -> None:
        """A server may not park a daily cycle for an hour by asking nicely."""
        sleep = RecordingSleep()
        served = {"count": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            served["count"] += 1
            if served["count"] == 1:
                return httpx.Response(503, json={}, headers={"retry-after": "3600"})
            return httpx.Response(200, json=_manifest_body())

        with _client(httpx.MockTransport(handler), validator_keypair, sleep=sleep) as db:
            db.get_snapshot(DAY)
        assert sleep.calls == [0.5]

    def test_a_transport_error_is_not_retried(self, validator_keypair: Keypair) -> None:
        """A reset says nothing about whether the request was processed."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            raise httpx.ConnectError("reset")

        with _client(httpx.MockTransport(handler), validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.get_snapshot(DAY)
        assert len(seen) == 1
        assert "transport" in str(caught.value)


class TestErrorMapping:
    def test_snapshot_not_ready_maps_to_its_own_error(self, validator_keypair: Keypair) -> None:
        transport, _seen = _scripted(
            [(409, _error_body(DbErrorCode.SNAPSHOT_NOT_READY.value, retry_after_seconds=300))]
        )
        with _client(transport, validator_keypair) as db:
            with pytest.raises(SnapshotNotReadyError) as caught:
                db.get_snapshot(DAY)
        assert "retry after 300s" in str(caught.value)

    def test_embargo_maps_to_its_own_error(self, validator_keypair: Keypair) -> None:
        transport, _seen = _scripted(
            [(403, _error_body(DbErrorCode.EMBARGOED.value, available_at=CYCLE_AT))]
        )
        with _client(transport, validator_keypair) as db:
            with pytest.raises(EmbargoedError) as caught:
                db.get_snapshot(DAY)
        assert str(CYCLE_AT) in str(caught.value)

    @pytest.mark.parametrize(
        "code",
        [
            DbErrorCode.AUTH_MALFORMED.value,
            DbErrorCode.AUTH_INVALID.value,
            DbErrorCode.AUTH_EXPIRED.value,
            DbErrorCode.AUTH_REPLAYED.value,
            DbErrorCode.NOT_AUTHORIZED.value,
        ],
    )
    def test_every_auth_code_maps_to_not_authorized(
        self, code: str, validator_keypair: Keypair
    ) -> None:
        transport, _seen = _scripted([(401, _error_body(code))])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(NotAuthorizedError) as caught:
                db.get_snapshot(DAY)
        assert caught.value.error_code == code

    def test_a_403_with_no_parseable_body_still_maps_to_not_authorized(
        self, validator_keypair: Keypair
    ) -> None:
        """Status is the fallback when a proxy answers instead of the service."""
        transport, _seen = _scripted([(403, "<html>forbidden</html>")])
        with _client(transport, validator_keypair) as db, pytest.raises(NotAuthorizedError):
            db.get_snapshot(DAY)

    def test_an_unrecognised_code_stays_a_generic_db_error(
        self, validator_keypair: Keypair
    ) -> None:
        transport, _seen = _scripted([(418, _error_body("db.something_new"))])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.get_snapshot(DAY)
        assert type(caught.value) is DbLayerError
        assert caught.value.error_code == "db.something_new"

    def test_a_non_json_success_body_is_reported_clearly(self, validator_keypair: Keypair) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        with _client(httpx.MockTransport(handler), validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.get_snapshot(DAY)
        assert "not JSON" in str(caught.value)

    def test_a_response_with_an_unexpected_field_is_refused(
        self, validator_keypair: Keypair
    ) -> None:
        """Silently ignoring it would mean hashing bytes we did not understand."""
        transport, _seen = _scripted([(200, _manifest_body(surprise="value"))])
        with _client(transport, validator_keypair) as db, pytest.raises(DbLayerError):
            db.get_snapshot(DAY)


class TestSignedPath:
    def test_the_signed_path_includes_a_base_path_prefix(self, validator_keypair: Keypair) -> None:
        transport, seen = _scripted([(200, _manifest_body())])
        with _client(transport, validator_keypair, base_url="https://db.invalid/api") as db:
            db.get_snapshot(DAY)
        assert seen[0].url.raw_path == b"/api/v2/snapshot/2026-08-05"

    def test_the_signed_path_includes_the_query_string(self, validator_keypair: Keypair) -> None:
        transport, seen = _scripted(
            [(200, {"date": DAY.isoformat(), "items": [], "next_cursor": None, "total_count": 0})]
        )
        with _client(transport, validator_keypair) as db:
            db.get_test_page(DAY, cursor="abc-_1", limit=50)
        assert seen[0].url.raw_path == b"/v2/dataset/2026-08-05/test?cursor=abc-_1&limit=50"

    def test_a_post_declares_its_content_type(
        self, validator_keypair: Keypair, signed_submission: EvaluationSubmission
    ) -> None:
        transport, seen = _scripted(
            [(201, {"evaluation_id": "x", "accepted_at": CYCLE_AT, "duplicate": False})]
        )
        with _client(transport, validator_keypair) as db:
            db.submit_evaluation(signed_submission)
        assert seen[0].headers["content-type"] == "application/json"
        assert seen[0].headers[auth.HEADER_BODY_SHA256] == auth.body_sha256(seen[0].content)

    def test_redirects_are_not_followed(self, validator_keypair: Keypair) -> None:
        """Following one would send a signature for a path we did not call."""
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(302, headers={"location": "https://elsewhere.invalid/v2"})

        with _client(httpx.MockTransport(handler), validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.get_snapshot(DAY)
        assert len(seen) == 1
        assert "redirected" in str(caught.value)


class TestArgumentGuards:
    def test_a_cursor_outside_the_base64url_alphabet_is_refused(
        self, validator_keypair: Keypair
    ) -> None:
        transport, _seen = _scripted([(200, _manifest_body())])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError):
                db.get_test_page(DAY, cursor="a b&limit=1")

    @pytest.mark.parametrize("limit", [0, 1001])
    def test_an_out_of_range_limit_is_refused(self, limit: int, validator_keypair: Keypair) -> None:
        transport, _seen = _scripted([(200, _manifest_body())])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError):
                db.get_test_page(DAY, limit=limit)

    def test_an_unsigned_evaluation_never_reaches_the_wire(
        self, validator_keypair: Keypair, submission: EvaluationSubmission
    ) -> None:
        transport, seen = _scripted([(201, {})])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError):
                db.submit_evaluation(submission)
        assert seen == []

    def test_negative_retries_are_refused(self, validator_keypair: Keypair) -> None:
        transport, _seen = _scripted([(200, _manifest_body())])
        with pytest.raises(DbLayerError):
            _client(transport, validator_keypair, max_retries=-1)


class TestPaginationGuards:
    def test_a_cursor_that_does_not_advance_is_refused(self, validator_keypair: Keypair) -> None:
        page = {
            "date": DAY.isoformat(),
            "items": [],
            "next_cursor": "stuck",
            "total_count": 1,
        }
        transport, _seen = _scripted([(200, page)])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_test_content(DAY)
        assert "not advancing" in str(caught.value)

    def test_a_page_for_the_wrong_day_is_refused(self, validator_keypair: Keypair) -> None:
        page = {
            "date": (DAY + dt.timedelta(days=1)).isoformat(),
            "items": [],
            "next_cursor": None,
            "total_count": 0,
        }
        transport, _seen = _scripted([(200, page)])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError) as caught:
                db.fetch_test_content(DAY)
        assert "was served" in str(caught.value)

    def test_a_cursor_outside_the_alphabet_from_the_server_is_refused(
        self, validator_keypair: Keypair
    ) -> None:
        page = {
            "date": DAY.isoformat(),
            "items": [],
            "next_cursor": "has spaces",
            "total_count": 2,
        }
        transport, _seen = _scripted([(200, page)])
        with _client(transport, validator_keypair) as db:
            with pytest.raises(DbLayerError):
                db.fetch_test_content(DAY)


class TestFromConfig:
    def test_it_takes_the_url_netuid_and_retry_budget_from_config(
        self, validator_keypair: Keypair
    ) -> None:
        config = Config(
            chain=ChainConfig(network=ChainNetwork.LOCAL, netuid=NETUID),
            wallet=WalletConfig(name="w", hotkey="h"),
            db=DbLayerConfig(base_url=BASE_URL, max_retries=0),
        )
        transport, seen = _scripted([(503, {})])
        with DbClient.from_config(config, validator_keypair, transport=transport) as db:
            with pytest.raises(DbLayerError):
                db.get_snapshot(DAY)
        assert len(seen) == 1
        assert seen[0].headers[auth.HEADER_NETUID] == str(NETUID)


@pytest.fixture
def submission(validator_keypair: Keypair) -> EvaluationSubmission:
    return EvaluationSubmission(
        date=DAY,
        netuid=NETUID,
        validator_hotkey=validator_keypair.ss58_address,
        scoring_version="prometheon-scoring/2.0.0",
        policy_version="2026-08-07",
        snapshot_content_hash="a" * 64,
        corpus_content_hash="b" * 64,
        labelled_test_count=1,
        labelled_production_count=1,
        results=(
            MinerResult(
                hotkey=validator_keypair.ss58_address,
                uid=0,
                weight=1,
                dataset_submitted_count=1,
                dataset_valid_count=1,
                dataset_score_micro=1_000_000,
                model_status=ModelStatus.NOT_COMMITTED,
                model_items_scored=0,
                model_correct_count=0,
                raw_accuracy_bp=0,
                mean_total_tokens_milli=0,
                efficiency_penalty_bp=0,
                model_score_bp=0,
            ),
        ),
        burned_weight=0,
        started_at=CYCLE_AT,
        completed_at=CYCLE_AT + 1,
    )


@pytest.fixture
def signed_submission(
    validator_keypair: Keypair, submission: EvaluationSubmission
) -> EvaluationSubmission:
    return auth.sign_evaluation(validator_keypair, submission)


def _day_routes(
    *,
    snapshot_version: str = SNAPSHOT_VERSION,
    miners_date: dt.date | None = None,
    models_date: dt.date | None = None,
    frozen_at: int = 1_700_000_000,
) -> httpx.MockTransport:
    """Route a whole `fetch_day` over an empty but internally consistent day."""
    content_hash = snapshot_content_hash(
        date=DAY,
        test_items=(),
        production_items=(),
        eligible_miners=(),
        model_commitments=(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/snapshot/" in path:
            return httpx.Response(
                200,
                json=_manifest_body(snapshot_version=snapshot_version, content_hash=content_hash),
            )
        if path.endswith("/test"):
            return httpx.Response(
                200, json={"date": DAY.isoformat(), "items": [], "total_count": 0}
            )
        if path.endswith("/production"):
            return httpx.Response(
                200, json={"date": DAY.isoformat(), "items": [], "total_count": 0}
            )
        if "/eligible-miners/" in path:
            return httpx.Response(
                200, json={"date": (miners_date or DAY).isoformat(), "miners": []}
            )
        if "/models/" in path:
            return httpx.Response(
                200,
                json={
                    "date": (models_date or DAY).isoformat(),
                    "frozen_at": frozen_at,
                    "commitments": [],
                },
            )
        raise AssertionError(f"unrouted path {path}")

    return httpx.MockTransport(handler)


class TestASnapshotIsCheckedNotJustHashed:
    """The content hash is computed over what was served, so it cannot notice
    that the wrong thing was served consistently."""

    def test_a_mismatched_snapshot_version_names_the_skew(self, validator_keypair: Keypair) -> None:
        """Otherwise a version bump surfaces as "content hash mismatch".

        The hash preimage folds in the client's own constant, so a served
        version this build does not implement looks like data corruption to an
        operator when the real cause is a client upgrade.
        """
        with _client(
            _day_routes(snapshot_version="prometheon-snapshot/9"), validator_keypair
        ) as db:
            with pytest.raises(DbLayerError, match=r"snapshot-version|version"):
                db.fetch_day(DAY)

    def test_the_freeze_timestamp_survives_to_the_caller(self, validator_keypair: Keypair) -> None:
        """It is the only evidence the promised freeze happened."""
        with _client(_day_routes(), validator_keypair) as db:
            assert db.fetch_day(DAY).model_commitments_frozen_at > 0

    def test_a_collection_stamped_for_another_day_is_refused(
        self, validator_keypair: Keypair
    ) -> None:
        with _client(_day_routes(miners_date=DAY - dt.timedelta(days=1)), validator_keypair) as db:
            with pytest.raises(DbLayerError, match="stamped"):
                db.fetch_day(DAY)
