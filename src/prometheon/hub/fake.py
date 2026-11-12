"""An in-memory hub that serves the real contract.

The hub itself lives in a separate private repository. This fake is what lets
the validator path be written and tested against the contract before it ships,
and what the two implementations are compared against afterwards.

It is a fake, not a mock. It signs epochs with a real Ed25519 key, verifies
request envelopes and record signatures with the real verifiers, serves gzip
files whose hashes match the epoch, expires presigned URLs, and applies the
two-day embargo on the public corpus. A test that drives
:class:`~prometheon.hub.client.HubClient` through it exercises both sides of
the wire.

Use it as an ``httpx.MockTransport``::

    hub = FakeHub(netuid=1)
    hub.register(validator_hotkey, CallerRole.VALIDATOR)
    hub.publish_day(day, miners=[...], models=[...], test_lines=[...], ...)
    client = HubClient(base_url="https://hub.invalid", netuid=1,
                       epoch_keys=hub.epoch_keys, keypair=validator,
                       transport=hub.transport)
"""

from __future__ import annotations

import base64
import datetime as dt
import gzip
import hashlib
import json
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final

import httpx
from bittensor_wallet import Keypair
from pydantic import ValidationError

from prometheon.errors import HubError, SignatureError
from prometheon.hub.auth import (
    AUTH_HEADERS,
    CallerRole,
    InMemoryNonceStore,
    StaticHotkeyDirectory,
    require_validator,
    verify_evaluation,
    verify_request,
)
from prometheon.hub.epoch import epoch_signing_bytes
from prometheon.hub.models import EPOCH_SCHEMA, EvaluationSubmission, HubErrorCode
from prometheon.version import SCORING_VERSION

#: RFC 8032's first test key. Public, and only ever used by tests.
TEST_SIGNING_SEED: Final[str] = "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"

#: How long a presigned URL lives.
URL_LIFETIME_SECONDS: Final[int] = 3600

#: Days a day's test content stays out of the public corpus.
EMBARGO_DAYS: Final[int] = 2

#: Fixed page size of the public corpus.
CORPUS_PAGE_SIZE: Final[int] = 50

_STORAGE_HOST: Final[str] = "storage.invalid"
_DAY_RE: Final[str] = r"\d{4}-\d{2}-\d{2}"
_SS58: Final[str] = r"[1-9A-HJ-NP-Za-km-z]{46,48}"


def _unix_now() -> int:
    return int(time.time())


def _midnight(day: dt.date) -> int:
    return int(dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc).timestamp())


class _ReplyError(HubError):
    """An error the fake renders as the contract's error body."""

    def __init__(
        self, message: str, *, status: int, code: HubErrorCode, detail: dict[str, int] | None = None
    ) -> None:
        super().__init__(message, status_code=status, error_code=code.value)
        self.detail = detail or {}


def gzip_lines(lines: Sequence[Mapping[str, Any] | bytes]) -> bytes:
    """JSONL, gzipped deterministically. Raw ``bytes`` entries pass through as lines."""
    rendered = [
        line if isinstance(line, bytes) else json.dumps(line, ensure_ascii=False).encode("utf-8")
        for line in lines
    ]
    body = b"".join(line + b"\n" for line in rendered)
    return gzip.compress(body, mtime=0)


@dataclass
class _Day:
    raw_epoch: dict[str, Any]
    files: dict[str, bytes]
    test_lines: list[Mapping[str, Any] | bytes]


@dataclass
class FakeHub:
    """An in-memory implementation of the hub routes a validator uses."""

    netuid: int
    now: Callable[[], int] = _unix_now
    key_id: str = "hub-test"
    signing_seed: str = TEST_SIGNING_SEED
    days: dict[dt.date, _Day] = field(default_factory=dict)
    #: Days the fake answers 409 for, as if still building.
    not_ready: set[dt.date] = field(default_factory=set)
    results: dict[tuple[dt.date, str], bytes] = field(default_factory=dict)
    credits: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: ``(method, target)`` for every request, signed or not.
    request_log: list[tuple[str, str]] = field(default_factory=list)
    #: Where presigned URLs point. The default never resolves, which suits a
    #: MockTransport; :func:`serve` points it at the server itself.
    storage_base: str = f"https://{_STORAGE_HOST}"
    #: The owner validator, whose record the copy-mode route serves.
    owner_validator: str | None = None

    def __post_init__(self) -> None:
        self.directory = StaticHotkeyDirectory({}, netuid=self.netuid)
        self.nonces = InMemoryNonceStore()
        # The stub types the seed as bytes; the binding only accepts hex text.
        self._signer = Keypair.create_from_seed(self.signing_seed, crypto_type=0)  # type: ignore[arg-type]
        self._urls: dict[str, tuple[dt.date, str, int]] = {}

    # -- setup ---------------------------------------------------------------

    @property
    def public_key(self) -> str:
        public_key = self._signer.public_key
        assert public_key is not None
        return bytes(public_key).hex()

    @property
    def epoch_keys(self) -> dict[str, str]:
        """What a validator pins in ``[hub.epoch_keys]``."""
        return {self.key_id: self.public_key}

    def register(self, hotkey: str, role: CallerRole) -> None:
        self.directory.set_role(hotkey, role)

    def publish_day(
        self,
        day: dt.date,
        *,
        miners: Sequence[Mapping[str, Any]],
        models: Sequence[Mapping[str, Any]] = (),
        test_lines: Sequence[Mapping[str, Any] | bytes] = (),
        production_lines: Sequence[Mapping[str, Any] | bytes] = (),
        policy_version: str = "2026-08-07",
        policy_sha256: str = "0" * 64,
        frozen_block: int = 1_000,
        scoring_version: str = SCORING_VERSION,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build, sign and store one day's epoch and files. Returns the raw epoch."""
        test_file = gzip_lines(test_lines)
        production_file = gzip_lines(production_lines)
        unsigned: dict[str, Any] = {
            "schema": EPOCH_SCHEMA,
            "netuid": self.netuid,
            "day": day.isoformat(),
            "frozen_block": frozen_block,
            "scoring_version": scoring_version,
            "policy": {"version": policy_version, "sha256": policy_sha256},
            "miners": [dict(miner) for miner in miners],
            "models": [dict(model) for model in models],
            "files": {
                "test": {"sha256": hashlib.sha256(test_file).hexdigest(), "count": len(test_lines)},
                "production": {
                    "sha256": hashlib.sha256(production_file).hexdigest(),
                    "count": len(production_lines),
                },
            },
            "key_id": self.key_id,
            **dict(extra or {}),
        }
        raw = {**unsigned, "signature": self.sign(unsigned)}
        self.days[day] = _Day(
            raw_epoch=raw,
            files={"test": test_file, "production": production_file},
            test_lines=list(test_lines),
        )
        return raw

    def sign(self, unsigned: Mapping[str, Any]) -> str:
        return "0x" + bytes(self._signer.sign(epoch_signing_bytes(unsigned))).hex()

    def replace_file(self, day: dt.date, name: str, data: bytes) -> None:
        """Serve different bytes than the epoch was signed over."""
        self.days[day].files[name] = data

    # -- transport -----------------------------------------------------------

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        target = request.url.raw_path.decode("ascii")
        self.request_log.append((request.method, target))
        try:
            if request.url.host == _STORAGE_HOST or request.url.path.startswith("/storage/"):
                return self._download(request)
            return self._serve(request, target)
        except _ReplyError as reply:
            body: dict[str, Any] = {"code": reply.error_code, "message": str(reply)}
            body.update(reply.detail)
            return httpx.Response(reply.status_code or 500, json={"error": body})
        except HubError as exc:
            return httpx.Response(
                exc.status_code or 500,
                json={"error": {"code": exc.error_code or "db.internal", "message": str(exc)}},
            )

    def _serve(self, request: httpx.Request, target: str) -> httpx.Response:
        path = target.split("?", 1)[0]
        marker = path.find("/api/v1/")
        if marker < 0:
            raise _ReplyError(f"no such route: {path}", status=404, code=HubErrorCode.NOT_FOUND)
        route = path[marker:]

        if route == "/api/v1/validator/epoch" and request.method == "POST":
            return self._epoch(request, target)
        if route == "/api/v1/validator/weights" and request.method == "POST":
            return self._weights(request, target)
        if route == "/api/v1/results" and request.method == "POST":
            return self._post_result(request, target)
        if request.method == "GET":
            if match := re.fullmatch(rf"/api/v1/results/({_DAY_RE})/({_SS58})", route):
                stored = self.results.get((_day(match.group(1)), match.group(2)))
                if stored is None:
                    raise _ReplyError("no such result", status=404, code=HubErrorCode.NOT_FOUND)
                return httpx.Response(200, content=stored)
            if match := re.fullmatch(rf"/api/v1/epochs/({_DAY_RE})", route):
                day = self._built(_day(match.group(1)))
                return httpx.Response(200, json=day.raw_epoch)
            if match := re.fullmatch(rf"/api/v1/credits/({_SS58})", route):
                address = match.group(1)
                payload = self.credits.get(
                    address, {"coldkey": address, "balance": 0, "entries": []}
                )
                return httpx.Response(200, json=payload)
            if route == "/api/v1/corpus":
                return self._corpus(request)
        raise _ReplyError(
            f"no such route: {request.method} {route}", status=404, code=HubErrorCode.NOT_FOUND
        )

    # -- routes --------------------------------------------------------------

    def _authenticate(self, request: httpx.Request, target: str) -> Any:
        return verify_request(
            method=request.method,
            path=target,
            body=request.content,
            headers=dict(request.headers),
            netuid=self.netuid,
            directory=self.directory,
            nonce_store=self.nonces,
            now=self.now(),
        )

    def _epoch(self, request: httpx.Request, target: str) -> httpx.Response:
        require_validator(self._authenticate(request, target))
        day = _requested_day(request)
        stored = self._built(day)
        expires = self.now() + URL_LIFETIME_SECONDS
        urls = {name: self._presign(day, name, expires) for name in ("test", "production")}
        return httpx.Response(
            200, json={"epoch": stored.raw_epoch, "urls": urls, "urls_expire_at": expires}
        )

    def _weights(self, request: httpx.Request, target: str) -> httpx.Response:
        """The owner validator's record for a day, for validators in copy mode.

        Served byte for byte as it was posted, inside ``{"record": ...}``, so its
        signature still verifies. 409 while the day can still get one, 404 for a
        day that never will.
        """
        require_validator(self._authenticate(request, target))
        day = _requested_day(request)
        stored = (
            self.results.get((day, self.owner_validator))
            if self.owner_validator is not None
            else None
        )
        if stored is not None:
            return httpx.Response(
                200,
                content=b'{"record":' + stored + b"}",
                headers={"content-type": "application/json"},
            )
        if self.owner_validator is not None and (day in self.days or day in self.not_ready):
            raise _ReplyError(
                f"the owner validator has not posted {day.isoformat()} yet",
                status=409,
                code=HubErrorCode.SNAPSHOT_NOT_READY,
                detail={"retry_after_seconds": 600},
            )
        raise _ReplyError(
            f"there is no owner validator record for {day.isoformat()}, and there will not be",
            status=404,
            code=HubErrorCode.NOT_FOUND,
        )

    def _post_result(self, request: httpx.Request, target: str) -> httpx.Response:
        # The envelope is optional on this route, but verified when sent.
        lowered = {key.lower() for key in request.headers}
        if any(header in lowered for header in AUTH_HEADERS):
            self._authenticate(request, target)
        try:
            submission = EvaluationSubmission.model_validate_json(request.content)
        except ValidationError as exc:
            raise _ReplyError(
                f"record is not acceptable: {exc}", status=422, code=HubErrorCode.INVALID_PAYLOAD
            ) from exc
        if (
            self.directory.role_for(submission.validator_hotkey, self.netuid)
            is not CallerRole.VALIDATOR
        ):
            raise _ReplyError(
                "the record's validator holds no permit",
                status=403,
                code=HubErrorCode.NOT_AUTHORIZED,
            )
        if submission.netuid != self.netuid:
            raise _ReplyError(
                "record is for another netuid", status=400, code=HubErrorCode.BAD_REQUEST
            )
        if submission.date not in self.days:
            raise _ReplyError(
                "no epoch exists for that day", status=400, code=HubErrorCode.BAD_REQUEST
            )
        try:
            verify_evaluation(submission)
        except SignatureError as exc:
            raise _ReplyError(
                f"record signature does not verify: {exc}",
                status=422,
                code=HubErrorCode.INVALID_PAYLOAD,
            ) from exc

        key = (submission.date, submission.validator_hotkey)
        existing = self.results.get(key)
        receipt = {"evaluation_id": f"{key[0].isoformat()}:{key[1]}", "accepted_at": self.now()}
        if existing is not None:
            if existing != request.content:
                raise _ReplyError(
                    "a different record is already stored for this day and validator",
                    status=409,
                    code=HubErrorCode.DUPLICATE_EVALUATION,
                )
            return httpx.Response(200, json={**receipt, "duplicate": True})
        self.results[key] = request.content
        return httpx.Response(201, json={**receipt, "duplicate": False})

    def _corpus(self, request: httpx.Request) -> httpx.Response:
        """The public, embargoed, paged view of test content. No author field."""
        try:
            day = _day(request.url.params["day"])
        except KeyError as exc:
            raise _ReplyError("day is required", status=400, code=HubErrorCode.BAD_REQUEST) from exc
        available_at = _midnight(day + dt.timedelta(days=EMBARGO_DAYS))
        if self.now() < available_at:
            raise _ReplyError(
                "test content is public two days later",
                status=403,
                code=HubErrorCode.EMBARGOED,
                detail={"available_at": available_at},
            )
        lines = [line for line in self._built(day).test_lines if not isinstance(line, bytes)]
        offset = _cursor_offset(request.url.params.get("cursor"))
        page = lines[offset : offset + CORPUS_PAGE_SIZE]
        following = offset + CORPUS_PAGE_SIZE
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        key: line.get(key)
                        for key in ("id", "body", "claimed_violating", "created_at")
                    }
                    for line in page
                ],
                "next_cursor": _cursor(following) if following < len(lines) else None,
            },
        )

    def _download(self, request: httpx.Request) -> httpx.Response:
        token = request.url.path.rsplit("/", 1)[-1]
        found = self._urls.get(token)
        if found is None:
            return httpx.Response(403, content=b"AccessDenied")
        day, name, expires = found
        if self.now() >= expires:
            return httpx.Response(403, content=b"Request has expired")
        return httpx.Response(200, content=self.days[day].files[name])

    # -- helpers -------------------------------------------------------------

    def _built(self, day: dt.date) -> _Day:
        if day in self.not_ready:
            raise _ReplyError(
                f"the epoch for {day.isoformat()} is still being built",
                status=409,
                code=HubErrorCode.SNAPSHOT_NOT_READY,
                detail={"retry_after_seconds": 300},
            )
        stored = self.days.get(day)
        if stored is None:
            raise _ReplyError(
                f"no epoch for {day.isoformat()}", status=404, code=HubErrorCode.NOT_FOUND
            )
        return stored

    def _presign(self, day: dt.date, name: str, expires: int) -> str:
        token = secrets.token_hex(16)
        self._urls[token] = (day, name, expires)
        base = self.storage_base.rstrip("/")
        prefix = "" if base.endswith("/storage") or base.endswith(_STORAGE_HOST) else "/storage"
        return f"{base}{prefix}/{day.isoformat()}/{name}/{token}"


def _requested_day(request: httpx.Request) -> dt.date:
    """The ``{"day": "YYYY-MM-DD"}`` body both validator routes take."""
    try:
        body = json.loads(request.content)
        return _day(body["day"])
    except (ValueError, KeyError, TypeError) as exc:
        raise _ReplyError(
            'body must be {"day": "YYYY-MM-DD"}', status=400, code=HubErrorCode.BAD_REQUEST
        ) from exc


def _day(raw: object) -> dt.date:
    try:
        return dt.date.fromisoformat(str(raw))
    except ValueError as exc:
        raise _ReplyError(
            f"{raw!r} is not a calendar date", status=400, code=HubErrorCode.BAD_REQUEST
        ) from exc


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode()).decode().rstrip("=")


def _cursor_offset(cursor: str | None) -> int:
    if cursor is None:
        return 0
    try:
        return int(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except ValueError as exc:
        raise _ReplyError("bad cursor", status=400, code=HubErrorCode.BAD_CURSOR) from exc


def serve(hub: FakeHub, *, host: str = "127.0.0.1", port: int = 8600) -> ThreadingHTTPServer:
    """Serve ``hub`` over plain HTTP on a background thread, for local runs.

    For a development chain on one machine, and for tests that want a real
    socket: no TLS, no limits. ``port=0`` takes any free port; read it back
    from ``server_address``. Presigned URLs point back at this server. Call
    ``shutdown()`` on the result to stop it.
    """

    class Handler(BaseHTTPRequestHandler):
        def _relay(self) -> None:
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            request = httpx.Request(
                self.command,
                f"{origin}{self.path}",
                headers=dict(self.headers.items()),
                content=body,
            )
            response = hub.handler(request)
            payload = response.content
            self.send_response(response.status_code)
            self.send_header(
                "content-type", response.headers.get("content-type", "application/json")
            )
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._relay()

        def do_POST(self) -> None:
            self._relay()

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = ThreadingHTTPServer((host, port), Handler)
    origin = f"http://{host}:{server.server_address[1]}"
    hub.storage_base = f"{origin}/storage"
    threading.Thread(target=server.serve_forever, name="fake-hub", daemon=True).start()
    return server


__all__ = [
    "CORPUS_PAGE_SIZE",
    "EMBARGO_DAYS",
    "TEST_SIGNING_SEED",
    "FakeHub",
    "gzip_lines",
    "serve",
]
