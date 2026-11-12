"""HTTP client for the hub.

One signed call fetches the day: ``POST /api/v1/validator/epoch``, under the
same hotkey envelope v2 used, returns the signed epoch and presigned URLs for
its two content files. The client verifies the epoch against the pinned key,
downloads the files, checks them against the epoch, and parses them. Nothing
the hub says is used before the signature over it has verified.

A second signed call serves copy mode: ``POST /api/v1/validator/weights``
returns the owner validator's record for a day, exactly as it was posted. The
caller verifies it; see :mod:`prometheon.scoring.copied`.

The other routes are public reads (a published result, an epoch without file
URLs, a credit balance) and the results upload, which is authenticated by the
record's own signature and also carries the envelope.

**Retries.** 429, 502, 503 and 504 are retried with backoff, every attempt with
a fresh nonce and ``issued_at``, because reusing an envelope is a replay the hub
must refuse. Transport errors are retried too, which v2 did not do: every call
here is safe to repeat. The epoch request and the reads change nothing, and the
results upload re-sends byte-identical bytes, which the hub answers with
``duplicate: true`` rather than storing twice.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, TypeVar

import httpx
from bittensor_wallet import Keypair
from pydantic import BaseModel, ValidationError

from prometheon.errors import (
    EpochMissingError,
    EpochNotReadyError,
    EpochVerificationError,
    HubAuthError,
    HubError,
    RecordMissingError,
)
from prometheon.hub import auth
from prometheon.hub.epoch import check_epoch_is_for, epoch_content_hash, verify_epoch
from prometheon.hub.feed import DayFeed, parse_feed, read_records
from prometheon.hub.models import (
    API_PREFIX,
    CreditBalance,
    Epoch,
    EpochUrls,
    ErrorResponse,
    EvaluationAccepted,
    EvaluationSubmission,
    HubErrorCode,
    is_ss58,
)

_T = TypeVar("_T", bound=BaseModel)

#: Statuses that mean the hub did not process the request and asked for a retry.
RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})

_BACKOFF_BASE_SECONDS: Final[float] = 0.5
_BACKOFF_CAP_SECONDS: Final[float] = 30.0
_MAX_RETRY_AFTER_SECONDS: Final[float] = 60.0

#: Ceiling on one downloaded content file. The archive is hashed as it streams,
#: so the cap bounds memory before the hash can be compared.
MAX_DOWNLOAD_BYTES: Final[int] = 1 << 30

_AUTH_CODES: Final[frozenset[str]] = frozenset(
    {
        HubErrorCode.AUTH_MALFORMED.value,
        HubErrorCode.AUTH_INVALID.value,
        HubErrorCode.AUTH_EXPIRED.value,
        HubErrorCode.AUTH_REPLAYED.value,
        HubErrorCode.NOT_AUTHORIZED.value,
    }
)


def _unix_now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class VerifiedEpoch:
    """An epoch whose signature has verified, with the raw form it was signed in."""

    epoch: Epoch
    raw: Mapping[str, Any]
    #: sha256 of the signed bytes; the name this epoch goes by in records.
    content_hash: str


@dataclass(frozen=True)
class EpochBundle:
    verified: VerifiedEpoch
    urls: EpochUrls
    urls_expire_at: int | None


@dataclass(frozen=True)
class DayData:
    """Everything a cycle reads from the hub, already checked."""

    verified: VerifiedEpoch
    feed: DayFeed

    @property
    def epoch(self) -> Epoch:
        return self.verified.epoch


class HubClient:
    """Talks to the hub, as a hotkey when it has one, anonymously otherwise."""

    def __init__(
        self,
        *,
        base_url: str,
        netuid: int,
        epoch_keys: Mapping[str, str],
        keypair: Keypair | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], int] = _unix_now,
        nonce_factory: Callable[[], str] = auth.new_nonce,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise HubError("max_retries must not be negative")
        self._netuid = netuid
        self._keys = dict(epoch_keys)
        self._keypair = keypair
        self._max_retries = max_retries
        self._now = now
        self._nonce_factory = nonce_factory
        self._sleep = sleep
        # A trailing slash makes joining append rather than replace the last
        # path segment, so a base URL with a path prefix keeps it (v2's B14).
        self._base = httpx.URL(base_url if base_url.endswith("/") else base_url + "/")
        self._http = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_seconds),
            # Hub routes are signed over their path, and a redirect would send a
            # signature for a path the request no longer has.
            follow_redirects=False,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> HubClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- the day -----------------------------------------------------------

    def request_epoch(self, day: dt.date) -> EpochBundle:
        """``POST /api/v1/validator/epoch``: the signed epoch and its file URLs."""
        body = json.dumps({"day": day.isoformat()}, separators=(",", ":")).encode("utf-8")
        response = self._send("POST", f"{API_PREFIX}/validator/epoch", body=body, signed=True)
        raw = _json(response)
        if not isinstance(raw, dict) or not isinstance(raw.get("epoch"), dict):
            raise EpochVerificationError("the epoch response carries no epoch object")
        verified = self._verify(raw["epoch"])
        try:
            urls = EpochUrls.model_validate(raw.get("urls"))
        except ValidationError as exc:
            raise HubError(f"the epoch response carries no usable file URLs: {exc}") from exc
        expires = raw.get("urls_expire_at")
        return EpochBundle(
            verified=verified,
            urls=urls,
            urls_expire_at=expires if isinstance(expires, int) else None,
        )

    def fetch_day(self, day: dt.date, *, scoring_version: str) -> DayData:
        """The day's verified epoch and its checked, parsed content."""
        bundle = self.request_epoch(day)
        epoch = bundle.verified.epoch
        check_epoch_is_for(epoch, day=day, netuid=self._netuid, scoring_version=scoring_version)
        test = read_records(self.download(bundle.urls.test), name="test", expected=epoch.files.test)
        production = read_records(
            self.download(bundle.urls.production),
            name="production",
            expected=epoch.files.production,
        )
        feed = parse_feed(
            test_records=test,
            production_records=production,
            day=day,
            miners={miner.hotkey: miner for miner in epoch.miners},
        )
        return DayData(verified=bundle.verified, feed=feed)

    def request_weights(self, day: dt.date) -> EvaluationSubmission:
        """``POST /api/v1/validator/weights``: the owner validator's record for ``day``.

        Validators only, signed like the epoch request. The record comes back
        exactly as the owner validator posted it, and nothing here trusts it:
        the caller checks who signed it before using a unit of it.

        Raises :class:`EpochNotReadyError` while the owner validator has not
        posted the day yet, which the scheduler retries without counting, and
        :class:`RecordMissingError` for a day that will never have one.
        """
        body = json.dumps({"day": day.isoformat()}, separators=(",", ":")).encode("utf-8")
        response = self._send("POST", f"{API_PREFIX}/validator/weights", body=body, signed=True)
        raw = _json(response)
        if not isinstance(raw, dict) or not isinstance(raw.get("record"), dict):
            raise HubError(
                "the weights response carries no record object", status_code=response.status_code
            )
        try:
            return EvaluationSubmission.model_validate(raw["record"])
        except ValidationError as exc:
            raise HubError(
                f"the hub returned a record this client cannot accept: {exc}",
                status_code=response.status_code,
            ) from exc

    def download(self, url: str) -> bytes:
        """One presigned file, unsigned, capped while it streams."""
        parsed = httpx.URL(url)
        if parsed.scheme != "https" and parsed.host not in ("127.0.0.1", "localhost"):
            raise HubError(f"refusing to download content over {parsed.scheme or 'no scheme'}")

        def attempt() -> httpx.Response:
            chunks: list[bytes] = []
            total = 0
            with self._http.stream("GET", url, follow_redirects=True) as response:
                if response.status_code >= 400:
                    return httpx.Response(response.status_code)
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise HubError(
                            f"a content file is larger than the {MAX_DOWNLOAD_BYTES} byte ceiling"
                        )
                    chunks.append(chunk)
                return httpx.Response(response.status_code, content=b"".join(chunks))

        response = self._with_retries(attempt, what="content download")
        if response.status_code != 200:
            raise HubError(
                f"downloading a content file failed with HTTP {response.status_code}; a "
                "presigned URL lasts an hour, so a retry fetches the epoch again",
                status_code=response.status_code,
            )
        return response.content

    # -- public reads ------------------------------------------------------

    def get_public_epoch(self, day: dt.date) -> VerifiedEpoch:
        """``GET /api/v1/epochs/{day}``, verified against the pinned key."""
        raw = _json(self._send("GET", f"{API_PREFIX}/epochs/{day.isoformat()}"))
        if isinstance(raw, dict) and isinstance(raw.get("epoch"), dict):
            raw = raw["epoch"]
        return self._verify(raw)

    def get_result(self, day: dt.date, validator_hotkey: str) -> EvaluationSubmission:
        """One validator's record, verbatim. The caller verifies its signature."""
        if not is_ss58(validator_hotkey):
            raise HubError(f"not an SS58 address: {validator_hotkey!r}")
        response = self._send("GET", f"{API_PREFIX}/results/{day.isoformat()}/{validator_hotkey}")
        return _parse(response, EvaluationSubmission)

    def get_credits(self, address: str) -> CreditBalance:
        """``GET /api/v1/credits/{coldkey_or_hotkey}``."""
        if not is_ss58(address):
            raise HubError(f"not an SS58 address: {address!r}")
        return _parse(self._send("GET", f"{API_PREFIX}/credits/{address}"), CreditBalance)

    # -- results -----------------------------------------------------------

    def submit_result(self, submission: EvaluationSubmission) -> EvaluationAccepted:
        """``POST /api/v1/results`` with a signed record.

        Send the same bytes again after a lost response, never a re-signed
        record: SR25519 signatures are randomised, so a re-sign is a different
        record, and the hub keeps the first one it stored.
        """
        if not submission.signature:
            raise HubError("the record is unsigned; sign it before submitting")
        body = submission.model_dump_json(exclude_none=False).encode("utf-8")
        return self.submit_result_bytes(body)

    def submit_result_bytes(self, body: bytes) -> EvaluationAccepted:
        response = self._send("POST", f"{API_PREFIX}/results", body=body, signed=True)
        return _parse(response, EvaluationAccepted)

    # -- transport ---------------------------------------------------------

    def _verify(self, raw: Any) -> VerifiedEpoch:
        epoch = verify_epoch(raw, keys=self._keys)
        return VerifiedEpoch(epoch=epoch, raw=raw, content_hash=epoch_content_hash(raw))

    def _send(
        self, method: str, relative: str, *, body: bytes | None = None, signed: bool = False
    ) -> httpx.Response:
        url = self._base.join(relative)
        signed_path = url.raw_path.decode("ascii")
        payload = body or b""
        if signed and self._keypair is None:
            raise HubError(f"{method} {signed_path} needs a hotkey to sign with")

        def attempt() -> httpx.Response:
            headers: dict[str, str] = {}
            if signed and self._keypair is not None:
                headers = auth.signed_headers(
                    self._keypair,
                    method=method,
                    path=signed_path,
                    body=payload,
                    netuid=self._netuid,
                    hotkey=str(self._keypair.ss58_address),
                    nonce=self._nonce_factory(),
                    issued_at=self._now(),
                )
            if body is not None:
                headers["content-type"] = "application/json"
            return self._http.request(method, url, headers=headers, content=payload)

        response = self._with_retries(attempt, what=f"{method} {signed_path}")
        if 300 <= response.status_code < 400:
            raise HubError(
                f"{method} {signed_path} was redirected to "
                f"{response.headers.get('location', 'an unnamed location')}; redirects are not "
                "followed because the signature covers the path",
                status_code=response.status_code,
            )
        if response.status_code >= 400:
            raise _error_for(response, method, signed_path)
        return response

    def _with_retries(self, attempt: Callable[[], httpx.Response], *, what: str) -> httpx.Response:
        tries = 0
        while True:
            try:
                response = attempt()
            except httpx.HTTPError as exc:
                if tries >= self._max_retries:
                    raise HubError(f"{what} failed in transport: {exc}") from exc
                self._sleep(_backoff_seconds(tries, None))
                tries += 1
                continue
            if response.status_code in RETRYABLE_STATUSES and tries < self._max_retries:
                self._sleep(_backoff_seconds(tries, response.headers.get("retry-after")))
                tries += 1
                continue
            return response


def _json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise HubError(
            f"the hub returned a body that is not JSON (HTTP {response.status_code})",
            status_code=response.status_code,
        ) from exc


def _parse(response: httpx.Response, model: type[_T]) -> _T:
    try:
        return model.model_validate(_json(response))
    except ValidationError as exc:
        raise HubError(
            f"the hub returned a {model.__name__} this client cannot accept: {exc}",
            status_code=response.status_code,
        ) from exc


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            seconds = int(retry_after)
        except ValueError:
            seconds = 0
        # Zero means "no wait" to a proxy and a tight loop to us, so it is
        # treated as absent, as is anything past our own ceiling.
        if 0 < seconds <= _MAX_RETRY_AFTER_SECONDS:
            return float(seconds)
    return min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2.0**attempt))


def _error_for(response: httpx.Response, method: str, path: str) -> HubError:
    """A non-2xx response as the typed error its code and status name."""
    status = response.status_code
    code: str | None = None
    message = f"{method} {path} failed with HTTP {status}"
    try:
        parsed = ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        parsed = None
    if parsed is not None:
        code = parsed.error.code
        message = f"{method} {path} failed with HTTP {status} ({code}): {parsed.error.message}"
        if parsed.error.retry_after_seconds is not None:
            message += f"; retry after {parsed.error.retry_after_seconds}s"

    if code == HubErrorCode.SNAPSHOT_NOT_READY.value or (code is None and status == 409):
        return EpochNotReadyError(message, status_code=status, error_code=code)
    if code in _AUTH_CODES or (code is None and status in (401, 403)):
        return HubAuthError(message, status_code=status, error_code=code)
    if status == 404 and path.endswith("/validator/epoch"):
        return EpochMissingError(message, status_code=status, error_code=code)
    if status == 404 and path.endswith("/validator/weights"):
        return RecordMissingError(message, status_code=status, error_code=code)
    return HubError(message, status_code=status, error_code=code)


__all__ = [
    "MAX_DOWNLOAD_BYTES",
    "RETRYABLE_STATUSES",
    "DayData",
    "EpochBundle",
    "HubClient",
    "VerifiedEpoch",
]
