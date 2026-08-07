"""Hotkey-signed request authentication for the subnet DB layer.

There are no API keys. A caller proves who it is by signing an envelope with
its hotkey, and the server checks that signature against the metagraph. A
deregistered hotkey loses access the moment the metagraph says so, with nobody
having to remember to revoke anything.

The envelope covers method, path, body digest, netuid, hotkey, nonce, and
issue time::

    {domain, method, path, body_sha256, netuid, hotkey, nonce, issued_at}

Each field closes a specific hole, and a server implementer who drops one will
not see anything break:

- ``method`` and ``path`` bind the signature to one route. Without them a
  signed read of yesterday's manifest is a signed read of anything.
- ``body_sha256`` binds the payload. Without it a captured POST envelope
  authenticates a substituted body.
- ``netuid`` stops an envelope signed for a testnet deployment from
  authenticating against mainnet.
- ``nonce`` and ``issued_at`` bound replay to a 300-second window and then to
  nothing at all.

**The server reconstructs ``method``, ``path``, and ``body_sha256`` from what it
actually received, never from the headers.** The headers exist so the client can
say what it signed; if they disagree with the request, the signature does not
verify and the request is rejected. Verifying against client-supplied values
instead would make all three fields decorative.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Protocol

from bittensor_wallet import Keypair
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from prometheon.dbclient.models import DbErrorCode, EvaluationSubmission
from prometheon.errors import NotAuthorizedError, SignatureError
from prometheon.security import signatures
from prometheon.security.canonical_json import DOMAIN_DB_REQUEST, DOMAIN_EVALUATION

HEADER_NETUID: Final[str] = "x-prometheon-netuid"
HEADER_HOTKEY: Final[str] = "x-prometheon-hotkey"
HEADER_NONCE: Final[str] = "x-prometheon-nonce"
HEADER_ISSUED_AT: Final[str] = "x-prometheon-issued-at"
HEADER_BODY_SHA256: Final[str] = "x-prometheon-body-sha256"
HEADER_SIGNATURE: Final[str] = "x-prometheon-signature"

AUTH_HEADERS: Final[tuple[str, ...]] = (
    HEADER_NETUID,
    HEADER_HOTKEY,
    HEADER_NONCE,
    HEADER_ISSUED_AT,
    HEADER_BODY_SHA256,
    HEADER_SIGNATURE,
)

#: How far an ``issued_at`` may sit from the server's clock, in either
#: direction. Both directions, because a client's clock can be fast.
AUTH_MAX_CLOCK_SKEW_SECONDS: Final[int] = 300

#: How long a spent nonce must stay remembered. Comfortably longer than twice
#: the skew window: a nonce forgotten while its envelope is still inside that
#: window is a nonce that can be replayed.
NONCE_RETENTION_SECONDS: Final[int] = 900

#: 16 random bytes as hex. Collision probability across a subnet's traffic is
#: negligible, and the store rejects one anyway.
NONCE_BYTES: Final[int] = 16

_NONCE_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: SHA-256 of the empty string: the body digest of every GET.
EMPTY_BODY_SHA256: Final[str] = hashlib.sha256(b"").hexdigest()


class CallerRole(str, Enum):
    """What the metagraph says a hotkey is.

    Only these two exist. A hotkey that is neither is not registered on this
    netuid and gets nothing.
    """

    VALIDATOR = "validator"
    MINER = "miner"


def new_nonce() -> str:
    return secrets.token_hex(NONCE_BYTES)


def body_sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class RequestEnvelope(BaseModel):
    """The object that gets signed. It travels as headers, never as JSON.

    Sending it as a header blob would mean the server parses attacker-supplied
    JSON to learn the method and path it is about to authorise. Discrete
    headers make it obvious that the server must use its own observed values.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    domain: str = Field(min_length=1, max_length=64)
    method: str = Field(min_length=3, max_length=16)
    path: str = Field(min_length=1, max_length=2048)
    body_sha256: str = Field(min_length=64, max_length=64)
    netuid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=64)
    nonce: str = Field(min_length=32, max_length=32)
    issued_at: int = Field(ge=0)

    def signing_payload(self) -> dict[str, Any]:
        """The exact object canonicalised under ``DOMAIN_DB_REQUEST``."""
        return {
            "body_sha256": self.body_sha256,
            "domain": self.domain,
            "hotkey": self.hotkey,
            "issued_at": self.issued_at,
            "method": self.method,
            "netuid": self.netuid,
            "nonce": self.nonce,
            "path": self.path,
        }


def build_envelope(
    *,
    method: str,
    path: str,
    body: bytes,
    netuid: int,
    hotkey: str,
    nonce: str,
    issued_at: int,
) -> RequestEnvelope:
    """Assemble the envelope for one request.

    ``path`` is the origin-form request target: everything after the authority,
    including the query string. That is exactly the bytes the server sees, so
    neither side has to agree on how to normalise a URL.
    """
    if not path.startswith("/"):
        raise NotAuthorizedError(
            f"signed path must be an origin-form request target, got {path!r}",
            error_code=DbErrorCode.AUTH_MALFORMED.value,
        )
    # The client side has the same hole as the server side: hand-raising for
    # one constraint and letting every other one escape as a bare
    # `ValidationError` means `signed_headers`, the documented one-call API,
    # carries an untyped failure mode. A `DbClient` built with an over-long
    # hotkey is the easy way to hit it.
    try:
        return RequestEnvelope(
            domain=DOMAIN_DB_REQUEST,
            method=method.upper(),
            path=path,
            body_sha256=body_sha256(body),
            netuid=netuid,
            hotkey=hotkey,
            nonce=nonce,
            issued_at=issued_at,
        )
    except ValidationError as exc:
        raise NotAuthorizedError(
            f"cannot sign a malformed request envelope: {_first_validation_problem(exc)}",
            error_code=DbErrorCode.AUTH_MALFORMED.value,
        ) from exc


def sign_envelope(keypair: Keypair, envelope: RequestEnvelope) -> str:
    return signatures.sign(keypair, DOMAIN_DB_REQUEST, envelope.signing_payload())


def envelope_headers(envelope: RequestEnvelope, signature: str) -> dict[str, str]:
    """Headers carrying an envelope. ``method`` and ``path`` are not among them.

    They are already on the request line, and a second copy is only an
    opportunity for the two to disagree.
    """
    return {
        HEADER_NETUID: str(envelope.netuid),
        HEADER_HOTKEY: envelope.hotkey,
        HEADER_NONCE: envelope.nonce,
        HEADER_ISSUED_AT: str(envelope.issued_at),
        HEADER_BODY_SHA256: envelope.body_sha256,
        HEADER_SIGNATURE: signature,
    }


def signed_headers(
    keypair: Keypair,
    *,
    method: str,
    path: str,
    body: bytes,
    netuid: int,
    hotkey: str,
    nonce: str,
    issued_at: int,
) -> dict[str, str]:
    """Build, sign, and render the headers for one request in one call."""
    envelope = build_envelope(
        method=method,
        path=path,
        body=body,
        netuid=netuid,
        hotkey=hotkey,
        nonce=nonce,
        issued_at=issued_at,
    )
    return envelope_headers(envelope, sign_envelope(keypair, envelope))


class NonceStore(Protocol):
    """Somewhere to remember spent nonces for :data:`NONCE_RETENTION_SECONDS`."""

    def claim(self, hotkey: str, nonce: str, issued_at: int, now: int) -> bool:
        """Record ``nonce`` for ``hotkey``; return ``False`` if already spent.

        Must be atomic. A check-then-set implementation lets two concurrent
        replays of the same envelope both pass.

        ``now`` is the *server's* clock and is what expiry must be measured
        against. ``issued_at`` is supplied by the caller and is therefore not a
        clock at all. See :class:`InMemoryNonceStore`.
        """


class InMemoryNonceStore:
    """Reference nonce store. Fine for one process, wrong for a fleet.

    A real deployment needs a shared, atomic store: ``SET NX`` with a TTL, or a
    unique index. Otherwise a replay simply goes to a different server and is
    accepted.
    """

    def __init__(
        self,
        retention_seconds: int = NONCE_RETENTION_SECONDS,
        *,
        max_clock_skew_seconds: int = AUTH_MAX_CLOCK_SKEW_SECONDS,
    ) -> None:
        # The invariant asserted where it can be checked, rather than left to a
        # sentence in the contract doc. A nonce forgotten while its envelope is
        # still inside the skew window is a nonce that can be replayed, and
        # both numbers were previously free parameters that nothing compared.
        minimum = 2 * max_clock_skew_seconds
        if retention_seconds <= minimum:
            raise NotAuthorizedError(
                f"nonce retention of {retention_seconds}s does not cover a "
                f"{max_clock_skew_seconds}s clock skew window: an envelope can be "
                f"replayed after its nonce is forgotten. Retention must exceed "
                f"{minimum}s",
                error_code=DbErrorCode.AUTH_MALFORMED.value,
            )
        self._retention = retention_seconds
        self._seen: dict[tuple[str, str], int] = {}

    def claim(self, hotkey: str, nonce: str, issued_at: int, now: int) -> bool:
        self._prune(now)
        key = (hotkey, nonce)
        if key in self._seen:
            return False
        # Stored against the server's clock too, so one caller's skew cannot
        # make its own nonce expire early or late.
        self._seen[key] = now
        return True

    def _prune(self, now: int) -> None:
        """Expire against the server's clock.

        Pruning against the incoming ``issued_at`` made expiry
        caller-controlled: a replayer choosing an ``issued_at`` at the far edge
        of the skew window advances the horizon and evicts the very record that
        would have caught the replay.
        """
        horizon = now - self._retention
        expired = [key for key, seen_at in self._seen.items() if seen_at < horizon]
        for key in expired:
            del self._seen[key]

    def __len__(self) -> int:
        return len(self._seen)


class HotkeyDirectory(Protocol):
    """The metagraph, as far as authentication is concerned."""

    def role_for(self, hotkey: str, netuid: int) -> CallerRole | None:
        """``None`` when the hotkey is not registered on ``netuid``."""


class StaticHotkeyDirectory:
    """A fixed hotkey-to-role map, for the fake and for tests."""

    def __init__(self, roles: Mapping[str, CallerRole], *, netuid: int) -> None:
        self._roles = dict(roles)
        self._netuid = netuid

    def role_for(self, hotkey: str, netuid: int) -> CallerRole | None:
        if netuid != self._netuid:
            return None
        return self._roles.get(hotkey)

    def set_role(self, hotkey: str, role: CallerRole | None) -> None:
        if role is None:
            self._roles.pop(hotkey, None)
        else:
            self._roles[hotkey] = role


@dataclass(frozen=True)
class AuthenticatedCaller:
    """Who the server decided is calling."""

    hotkey: str
    role: CallerRole
    envelope: RequestEnvelope


def _reject(message: str, code: DbErrorCode, status: int) -> NotAuthorizedError:
    return NotAuthorizedError(message, status_code=status, error_code=code.value)


def _first_validation_problem(exc: ValidationError) -> str:
    """One field and one reason, for an error that reaches an untrusted caller.

    Not the full pydantic report: that echoes submitted values back, and these
    values came from an unauthenticated request. The field name and the
    constraint are enough for an honest client to fix its call and tell an
    attacker nothing it did not already send.
    """
    errors = exc.errors()
    if not errors:
        return "unknown constraint violation"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "envelope"
    return f"{location}: {first.get('msg', 'invalid')}"


def verify_request(
    *,
    method: str,
    path: str,
    body: bytes,
    headers: Mapping[str, str],
    netuid: int,
    directory: HotkeyDirectory,
    nonce_store: NonceStore,
    now: int,
    max_clock_skew_seconds: int = AUTH_MAX_CLOCK_SKEW_SECONDS,
) -> AuthenticatedCaller:
    """Authenticate one request, or raise :class:`NotAuthorizedError`.

    ``method``, ``path``, and ``body`` must be what the server received, not
    what the client claimed.

    The order of the checks is part of the contract. In particular the nonce is
    claimed **after** the signature verifies: claiming first lets anyone who can
    see a nonce burn it by replaying an unsigned request, which turns a replay
    defence into a denial-of-service tool against the caller it protects.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    missing = [name for name in AUTH_HEADERS if not lowered.get(name)]
    if missing:
        raise _reject(
            f"missing authentication headers: {', '.join(sorted(missing))}",
            DbErrorCode.AUTH_MALFORMED,
            401,
        )

    hotkey = lowered[HEADER_HOTKEY]
    nonce = lowered[HEADER_NONCE]
    claimed_digest = lowered[HEADER_BODY_SHA256]
    signature = lowered[HEADER_SIGNATURE]

    try:
        claimed_netuid = int(lowered[HEADER_NETUID])
        issued_at = int(lowered[HEADER_ISSUED_AT])
    except ValueError as exc:
        raise _reject(
            f"netuid and issued-at headers must be integers: {exc}",
            DbErrorCode.AUTH_MALFORMED,
            401,
        ) from exc

    if not _NONCE_RE.match(nonce):
        raise _reject("nonce must be 32 lowercase hex characters", DbErrorCode.AUTH_MALFORMED, 401)
    if not _HEX64_RE.match(claimed_digest):
        raise _reject(
            "body digest must be 64 lowercase hex characters", DbErrorCode.AUTH_MALFORMED, 401
        )

    if claimed_netuid != netuid:
        raise _reject(
            f"envelope is for netuid {claimed_netuid}, this service serves {netuid}",
            DbErrorCode.NOT_AUTHORIZED,
            403,
        )

    skew = abs(now - issued_at)
    if skew > max_clock_skew_seconds:
        raise _reject(
            f"issued_at is {skew}s from server time, limit is {max_clock_skew_seconds}s",
            DbErrorCode.AUTH_EXPIRED,
            401,
        )

    actual_digest = body_sha256(body)
    if not secrets.compare_digest(claimed_digest, actual_digest):
        raise _reject("body digest does not match the received body", DbErrorCode.AUTH_INVALID, 401)

    # Every value here is attacker-controlled: they arrive as raw headers from
    # an as-yet-unauthenticated caller. Constructing the envelope runs pydantic
    # field constraints, and a violation raises `ValidationError`, which is not
    # a `PrometheonError`, so it was caught by nothing on the way out.
    #
    # A `X-Prometheon-Hotkey` header of 100 characters passes every check above
    # (presence, nonce regex, digest regex, netuid, skew, body digest) and then
    # fails `max_length=64` here. The server's own handler catches
    # `DbLayerError` and does not see it; the client's `_request` catches
    # `httpx.HTTPError` and does not see it either. The result is that an
    # unauthenticated caller can force an untyped crash instead of the
    # `401 db.auth_malformed` the contract specifies.
    try:
        envelope = RequestEnvelope(
            domain=DOMAIN_DB_REQUEST,
            method=method.upper(),
            path=path,
            body_sha256=actual_digest,
            netuid=netuid,
            hotkey=hotkey,
            nonce=nonce,
            issued_at=issued_at,
        )
    except ValidationError as exc:
        raise _reject(
            f"request envelope is malformed: {_first_validation_problem(exc)}",
            DbErrorCode.AUTH_MALFORMED,
            401,
        ) from exc

    try:
        signatures.verify(
            ss58_address=hotkey,
            signature_hex=signature,
            domain=DOMAIN_DB_REQUEST,
            payload=envelope.signing_payload(),
        )
    except SignatureError as exc:
        # Not echoing which part failed: a caller who cannot produce a valid
        # signature has no use for the difference, and an attacker does.
        raise _reject(
            f"request signature is not valid: {exc}", DbErrorCode.AUTH_INVALID, 401
        ) from exc

    if not nonce_store.claim(hotkey, nonce, issued_at, now):
        raise _reject("nonce has already been used", DbErrorCode.AUTH_REPLAYED, 401)

    role = directory.role_for(hotkey, netuid)
    if role is None:
        raise _reject(
            f"{hotkey} is not registered on netuid {netuid}", DbErrorCode.NOT_AUTHORIZED, 403
        )

    return AuthenticatedCaller(hotkey=hotkey, role=role, envelope=envelope)


def require_validator(caller: AuthenticatedCaller) -> None:
    """Gate an endpoint that only a validator may reach."""
    if caller.role is not CallerRole.VALIDATOR:
        raise _reject(
            f"{caller.hotkey} does not hold a validator permit on netuid {caller.envelope.netuid}",
            DbErrorCode.NOT_AUTHORIZED,
            403,
        )


def sign_evaluation(keypair: Keypair, submission: EvaluationSubmission) -> EvaluationSubmission:
    """Return a copy of ``submission`` carrying its signature."""
    unsigned = submission.model_copy(update={"signature": ""})
    signature = signatures.sign(keypair, DOMAIN_EVALUATION, unsigned.signing_payload())
    return submission.model_copy(update={"signature": signature})


def verify_evaluation(submission: EvaluationSubmission) -> None:
    """Check an evaluation's own signature against its ``validator_hotkey``.

    Separate from the request envelope: that authenticates the *call*, this
    authenticates the *record*. A stored evaluation outlives the connection it
    arrived on, and anyone re-reading it later needs to check it without
    trusting whoever handed it over.
    """
    unsigned = submission.model_copy(update={"signature": ""})
    signatures.verify(
        ss58_address=submission.validator_hotkey,
        signature_hex=submission.signature,
        domain=DOMAIN_EVALUATION,
        payload=unsigned.signing_payload(),
    )


__all__ = [
    "AUTH_HEADERS",
    "AUTH_MAX_CLOCK_SKEW_SECONDS",
    "EMPTY_BODY_SHA256",
    "HEADER_BODY_SHA256",
    "HEADER_HOTKEY",
    "HEADER_ISSUED_AT",
    "HEADER_NETUID",
    "HEADER_NONCE",
    "HEADER_SIGNATURE",
    "NONCE_BYTES",
    "NONCE_RETENTION_SECONDS",
    "AuthenticatedCaller",
    "CallerRole",
    "HotkeyDirectory",
    "InMemoryNonceStore",
    "NonceStore",
    "RequestEnvelope",
    "StaticHotkeyDirectory",
    "body_sha256",
    "build_envelope",
    "envelope_headers",
    "new_nonce",
    "require_validator",
    "sign_envelope",
    "sign_evaluation",
    "signed_headers",
    "verify_evaluation",
    "verify_request",
]
