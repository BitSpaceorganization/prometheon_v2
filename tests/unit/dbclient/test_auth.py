"""Behaviour of the signed request envelope.

Each test here corresponds to something the envelope exists to stop. If a field
can be changed in flight without the signature failing, that field is not
protecting anything, and these are the tests that would notice.
"""

from __future__ import annotations

import pytest
from bittensor_wallet import Keypair
from tests.unit.dbclient.factories import CYCLE_AT, DAY, NETUID

from prometheon.dbclient import auth
from prometheon.dbclient.models import DbErrorCode, EvaluationSubmission, MinerResult, ModelStatus
from prometheon.errors import NotAuthorizedError, SignatureVerificationError

pytestmark = pytest.mark.unit

PATH = "/v2/snapshot/2026-08-05"


def _directory(validator: str, miner: str) -> auth.StaticHotkeyDirectory:
    return auth.StaticHotkeyDirectory(
        {validator: auth.CallerRole.VALIDATOR, miner: auth.CallerRole.MINER}, netuid=NETUID
    )


def _headers(
    keypair: Keypair,
    *,
    method: str = "GET",
    path: str = PATH,
    body: bytes = b"",
    netuid: int = NETUID,
    issued_at: int = CYCLE_AT,
    nonce: str | None = None,
) -> dict[str, str]:
    return auth.signed_headers(
        keypair,
        method=method,
        path=path,
        body=body,
        netuid=netuid,
        hotkey=keypair.ss58_address,
        nonce=nonce or auth.new_nonce(),
        issued_at=issued_at,
    )


def _verify(
    headers: dict[str, str],
    *,
    directory: auth.StaticHotkeyDirectory,
    store: auth.NonceStore,
    method: str = "GET",
    path: str = PATH,
    body: bytes = b"",
    now: int = CYCLE_AT,
) -> auth.AuthenticatedCaller:
    return auth.verify_request(
        method=method,
        path=path,
        body=body,
        headers=headers,
        netuid=NETUID,
        directory=directory,
        nonce_store=store,
        now=now,
    )


@pytest.fixture
def directory(validator_keypair: Keypair, miner_keypair: Keypair) -> auth.StaticHotkeyDirectory:
    return _directory(validator_keypair.ss58_address, miner_keypair.ss58_address)


@pytest.fixture
def store() -> auth.InMemoryNonceStore:
    return auth.InMemoryNonceStore()


class TestHappyPath:
    def test_a_signed_request_authenticates(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        caller = _verify(_headers(validator_keypair), directory=directory, store=store)
        assert caller.hotkey == validator_keypair.ss58_address
        assert caller.role is auth.CallerRole.VALIDATOR

    def test_the_envelope_does_not_travel_as_method_or_path_headers(
        self, validator_keypair: Keypair
    ) -> None:
        """Only the request line may say what route is being called."""
        headers = _headers(validator_keypair)
        assert not any("path" in name or "method" in name for name in headers)

    def test_a_post_body_is_covered(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        body = b'{"hello":"world"}'
        headers = _headers(validator_keypair, method="POST", path="/v2/evaluations", body=body)
        caller = _verify(
            headers,
            directory=directory,
            store=store,
            method="POST",
            path="/v2/evaluations",
            body=body,
        )
        assert caller.envelope.body_sha256 == auth.body_sha256(body)


class TestTampering:
    def test_a_replayed_envelope_on_another_path_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, path="/v2/snapshot/2026-08-05")
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store, path="/v2/snapshot/2026-08-04")
        assert caught.value.error_code == DbErrorCode.AUTH_INVALID.value

    def test_a_changed_query_string_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        """The signed path is the whole request target, cursor included."""
        signed = "/v2/dataset/2026-08-05/test?cursor=abc"
        headers = _headers(validator_keypair, path=signed)
        with pytest.raises(NotAuthorizedError):
            _verify(
                headers,
                directory=directory,
                store=store,
                path="/v2/dataset/2026-08-05/test?cursor=xyz",
            )

    def test_a_changed_method_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, method="GET")
        with pytest.raises(NotAuthorizedError):
            _verify(headers, directory=directory, store=store, method="DELETE")

    def test_a_substituted_body_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, method="POST", body=b'{"a":1}')
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store, method="POST", body=b'{"a":2}')
        assert caught.value.error_code == DbErrorCode.AUTH_INVALID.value

    def test_a_body_digest_header_that_lies_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        """The server hashes what arrived; the header is only a claim about it."""
        body = b'{"a":1}'
        headers = _headers(validator_keypair, method="POST", body=body)
        headers[auth.HEADER_BODY_SHA256] = auth.body_sha256(b'{"a":2}')
        with pytest.raises(NotAuthorizedError):
            _verify(headers, directory=directory, store=store, method="POST", body=body)

    def test_another_hotkeys_signature_is_rejected(
        self,
        validator_keypair: Keypair,
        stranger_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(stranger_keypair)
        headers[auth.HEADER_HOTKEY] = validator_keypair.ss58_address
        with pytest.raises(NotAuthorizedError):
            _verify(headers, directory=directory, store=store)


class TestNetuidBinding:
    def test_an_envelope_for_another_netuid_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, netuid=NETUID + 1)
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.NOT_AUTHORIZED.value
        assert caught.value.status_code == 403


class TestClockWindow:
    @pytest.mark.parametrize("offset", [-301, 301])
    def test_an_envelope_outside_the_window_is_rejected(
        self,
        offset: int,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, issued_at=CYCLE_AT + offset)
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store, now=CYCLE_AT)
        assert caught.value.error_code == DbErrorCode.AUTH_EXPIRED.value

    @pytest.mark.parametrize("offset", [-300, 0, 300])
    def test_an_envelope_inside_the_window_is_accepted(
        self,
        offset: int,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair, issued_at=CYCLE_AT + offset)
        assert _verify(headers, directory=directory, store=store, now=CYCLE_AT)


class TestReplay:
    def test_the_same_envelope_twice_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair)
        _verify(headers, directory=directory, store=store)
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.AUTH_REPLAYED.value

    def test_a_nonce_is_not_burned_by_a_request_that_fails_to_verify(
        self,
        validator_keypair: Keypair,
        stranger_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        """Otherwise anyone who can see a nonce can lock its owner out of it."""
        nonce = auth.new_nonce()
        forged = _headers(stranger_keypair, nonce=nonce)
        forged[auth.HEADER_HOTKEY] = validator_keypair.ss58_address
        with pytest.raises(NotAuthorizedError):
            _verify(forged, directory=directory, store=store)
        assert len(store) == 0

        genuine = _headers(validator_keypair, nonce=nonce)
        assert _verify(genuine, directory=directory, store=store)

    def test_two_hotkeys_may_use_the_same_nonce(
        self,
        validator_keypair: Keypair,
        miner_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        nonce = auth.new_nonce()
        assert _verify(_headers(validator_keypair, nonce=nonce), directory=directory, store=store)
        assert _verify(_headers(miner_keypair, nonce=nonce), directory=directory, store=store)

    def test_spent_nonces_are_forgotten_after_the_retention_window(self) -> None:
        store = auth.InMemoryNonceStore(retention_seconds=900)
        assert store.claim("hk", "a" * 32, CYCLE_AT, CYCLE_AT)
        assert not store.claim("hk", "a" * 32, CYCLE_AT, CYCLE_AT)
        assert store.claim("hk", "b" * 32, CYCLE_AT + 901, CYCLE_AT + 901)
        # The old entry is gone, so the store cannot grow without bound.
        assert len(store) == 1

    def test_expiry_runs_on_the_server_clock_not_the_callers(self) -> None:
        """`issued_at` is supplied by the caller, so it is not a clock.

        Pruning against it let a replayer pick an `issued_at` at the far edge
        of the skew window, advance the horizon, and evict the very record that
        would have caught the replay.
        """
        store = auth.InMemoryNonceStore(retention_seconds=900)
        nonce = "a" * 32
        assert store.claim("hk", nonce, CYCLE_AT, CYCLE_AT)

        # A replay moments later, claiming an issued_at far in the future.
        assert not store.claim("hk", nonce, CYCLE_AT + 100_000, CYCLE_AT + 1)

    def test_a_retention_that_cannot_cover_the_skew_window_is_refused(self) -> None:
        """The invariant asserted where it can be checked, not just documented.

        A nonce forgotten while its envelope is still inside the skew window is
        a nonce that can be replayed, and both numbers were free parameters
        that nothing compared.
        """
        with pytest.raises(NotAuthorizedError, match="does not cover"):
            auth.InMemoryNonceStore(retention_seconds=300, max_clock_skew_seconds=300)

    def test_the_shipped_defaults_satisfy_the_invariant(self) -> None:
        assert auth.NONCE_RETENTION_SECONDS > 2 * auth.AUTH_MAX_CLOCK_SKEW_SECONDS
        auth.InMemoryNonceStore()


class TestMalformedHeaders:
    @pytest.mark.parametrize("header", list(auth.AUTH_HEADERS))
    def test_a_missing_header_is_rejected(
        self,
        header: str,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair)
        del headers[header]
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.AUTH_MALFORMED.value

    def test_a_non_integer_issued_at_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair)
        headers[auth.HEADER_ISSUED_AT] = "soon"
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.AUTH_MALFORMED.value

    def test_a_short_nonce_is_rejected(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = _headers(validator_keypair)
        headers[auth.HEADER_NONCE] = "abc"
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(headers, directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.AUTH_MALFORMED.value

    def test_headers_are_matched_case_insensitively(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        headers = {name.upper(): value for name, value in _headers(validator_keypair).items()}
        assert _verify(headers, directory=directory, store=store)

    def test_a_relative_path_cannot_be_signed(self, validator_keypair: Keypair) -> None:
        with pytest.raises(NotAuthorizedError):
            auth.build_envelope(
                method="GET",
                path="v2/snapshot/2026-08-05",
                body=b"",
                netuid=NETUID,
                hotkey=validator_keypair.ss58_address,
                nonce=auth.new_nonce(),
                issued_at=CYCLE_AT,
            )


class TestRegistration:
    def test_an_unregistered_hotkey_is_rejected(
        self,
        stranger_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        with pytest.raises(NotAuthorizedError) as caught:
            _verify(_headers(stranger_keypair), directory=directory, store=store)
        assert caught.value.error_code == DbErrorCode.NOT_AUTHORIZED.value

    def test_deregistering_revokes_access_with_no_credential_to_rotate(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        assert _verify(_headers(validator_keypair), directory=directory, store=store)
        directory.set_role(validator_keypair.ss58_address, None)
        with pytest.raises(NotAuthorizedError):
            _verify(_headers(validator_keypair), directory=directory, store=store)

    def test_a_miner_cannot_reach_a_validator_only_endpoint(
        self,
        miner_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.InMemoryNonceStore,
    ) -> None:
        caller = _verify(_headers(miner_keypair), directory=directory, store=store)
        with pytest.raises(NotAuthorizedError) as caught:
            auth.require_validator(caller)
        assert caught.value.status_code == 403


def _submission(hotkey: str) -> EvaluationSubmission:
    return EvaluationSubmission(
        date=DAY,
        netuid=NETUID,
        validator_hotkey=hotkey,
        scoring_version="prometheon-scoring/2.0.0",
        policy_version="2026-08-07",
        snapshot_content_hash="a" * 64,
        corpus_content_hash="b" * 64,
        labelled_test_count=1,
        labelled_production_count=2,
        results=(
            MinerResult(
                hotkey=hotkey,
                uid=1,
                weight=100,
                dataset_submitted_count=2,
                dataset_valid_count=1,
                dataset_score_micro=500_000,
                model_status=ModelStatus.EVALUATED,
                model_items_scored=10,
                model_correct_count=9,
                raw_accuracy_bp=9000,
                mean_total_tokens_milli=1000,
                efficiency_penalty_bp=0,
                model_score_bp=9000,
                model_rank=1,
            ),
        ),
        burned_weight=0,
        started_at=CYCLE_AT,
        completed_at=CYCLE_AT + 60,
    )


class TestEvaluationSignature:
    def test_a_signed_evaluation_verifies(self, validator_keypair: Keypair) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        auth.verify_evaluation(signed)

    def test_editing_a_published_result_breaks_it(self, validator_keypair: Keypair) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        edited = signed.model_copy(update={"burned_weight": 500})
        with pytest.raises(SignatureVerificationError):
            auth.verify_evaluation(edited)

    def test_claiming_another_validators_hotkey_breaks_it(
        self, validator_keypair: Keypair, stranger_keypair: Keypair
    ) -> None:
        signed = auth.sign_evaluation(
            validator_keypair, _submission(validator_keypair.ss58_address)
        )
        stolen = signed.model_copy(update={"validator_hotkey": stranger_keypair.ss58_address})
        with pytest.raises(SignatureVerificationError):
            auth.verify_evaluation(stolen)

    def test_signing_twice_is_stable(self, validator_keypair: Keypair) -> None:
        """The payload is built by hand, so it must not depend on field order."""
        submission = _submission(validator_keypair.ss58_address)
        first = auth.sign_evaluation(validator_keypair, submission)
        second = auth.sign_evaluation(validator_keypair, first)
        auth.verify_evaluation(second)
        assert first.signing_payload() == second.signing_payload()


class TestAMalformedHeaderIsA401NotACrash:
    """An unauthenticated caller must never be able to force an untyped error.

    Every value in the envelope arrives as a raw header from someone who has
    not yet proved anything. Constructing the envelope runs pydantic field
    constraints, and a `ValidationError` is not a `PrometheonError`, so it was
    caught by neither the server handler (`except DbLayerError`) nor the client
    (`except httpx.HTTPError`) and surfaced as a 500.
    """

    def test_an_over_long_hotkey_is_rejected_as_malformed(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.NonceStore,
    ) -> None:
        """Passes presence, nonce, digest, netuid and skew, then fails max_length=64."""
        headers = _headers(validator_keypair)
        headers[auth.HEADER_HOTKEY] = "5" + "G" * 100

        with pytest.raises(NotAuthorizedError) as excinfo:
            _verify(headers, directory=directory, store=store)

        assert excinfo.value.error_code == DbErrorCode.AUTH_MALFORMED.value
        assert excinfo.value.status_code == 401

    def test_an_over_long_path_is_rejected_as_malformed(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.NonceStore,
    ) -> None:
        with pytest.raises(NotAuthorizedError) as excinfo:
            _verify(
                _headers(validator_keypair),
                directory=directory,
                store=store,
                path="/v2/" + "z" * 3000,
            )

        assert excinfo.value.error_code == DbErrorCode.AUTH_MALFORMED.value

    def test_the_rejection_does_not_echo_the_submitted_value_back(
        self,
        validator_keypair: Keypair,
        directory: auth.StaticHotkeyDirectory,
        store: auth.NonceStore,
    ) -> None:
        """The value came from an unauthenticated request; naming the field is enough."""
        headers = _headers(validator_keypair)
        headers[auth.HEADER_HOTKEY] = "5" + "G" * 100

        with pytest.raises(NotAuthorizedError) as excinfo:
            _verify(headers, directory=directory, store=store)

        message = str(excinfo.value)
        assert "hotkey" in message
        assert "G" * 100 not in message

    def test_the_client_cannot_sign_a_malformed_envelope_either(self) -> None:
        """`signed_headers` is the documented one-call API and must be typed too."""
        with pytest.raises(NotAuthorizedError) as excinfo:
            auth.build_envelope(
                method="GET",
                path=PATH,
                body=b"",
                netuid=NETUID,
                hotkey="5" + "G" * 100,
                nonce=auth.new_nonce(),
                issued_at=CYCLE_AT,
            )

        assert excinfo.value.error_code == DbErrorCode.AUTH_MALFORMED.value
