"""An in-memory DB layer that serves the real contract.

The service this speaks for is a separate private system that does not exist
yet. This is what lets the validator path be written, tested, and reviewed
before it does. Once it does, it is what the two implementations are checked
against.

It is a fake and not a mock. It verifies signatures with the real verifier,
enforces the real nonce and clock rules, applies the real embargo, computes the
real content hash, and paginates with real cursors. A test that drives
:class:`~prometheon.dbclient.client.DbClient` through this exercises both sides
of the wire contract; a test built on stubbed responses would only check that
two pieces of the same test agree.

Use it as an ``httpx.MockTransport`` handler::

    fake = FakeDbLayer(netuid=42)
    fake.register(validator_hotkey, CallerRole.VALIDATOR)
    fake.seed_day(day, test_items=[...], production_items=[...])
    client = DbClient(base_url="https://db.invalid", netuid=42,
                      keypair=validator_keypair, transport=fake.transport)
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import httpx
from pydantic import BaseModel, ValidationError

from prometheon.dbclient.auth import (
    AuthenticatedCaller,
    CallerRole,
    InMemoryNonceStore,
    StaticHotkeyDirectory,
    require_validator,
    verify_evaluation,
    verify_request,
)
from prometheon.dbclient.models import (
    DEFAULT_PAGE_LIMIT,
    EMBARGO_DAYS,
    MAX_PAGE_LIMIT,
    PRODUCTION_ITEM_CAP,
    SNAPSHOT_VERSION,
    DbErrorCode,
    EligibleMiner,
    EligibleMinerSet,
    EvaluationAccepted,
    EvaluationSubmission,
    ModelCommitment,
    ModelCommitmentSet,
    ProductionContentItem,
    ProductionContentPage,
    SnapshotManifest,
    TestContentItem,
    TestContentPage,
    snapshot_content_hash,
)
from prometheon.errors import DbLayerError, SignatureError

_SNAPSHOT_RE: Final[re.Pattern[str]] = re.compile(r"^/v2/snapshot/(?P<date>\d{4}-\d{2}-\d{2})$")
_DATASET_RE: Final[re.Pattern[str]] = re.compile(
    r"^/v2/dataset/(?P<date>\d{4}-\d{2}-\d{2})/(?P<collection>test|production)$"
)
_MINERS_RE: Final[re.Pattern[str]] = re.compile(
    r"^/v2/eligible-miners/(?P<date>\d{4}-\d{2}-\d{2})$"
)
_MODELS_RE: Final[re.Pattern[str]] = re.compile(r"^/v2/models/(?P<date>\d{4}-\d{2}-\d{2})$")
_EVALUATIONS_PATH: Final[str] = "/v2/evaluations"

_SECONDS_PER_DAY: Final[int] = 86_400

#: What the fake tells a client to wait before asking for an unbuilt snapshot
#: again. Long enough that a validator retrying does not hammer the builder.
_SNAPSHOT_RETRY_AFTER_SECONDS: Final[int] = 300


def _unix_now() -> int:
    return int(time.time())


def _midnight_utc(day: dt.date) -> int:
    return int(dt.datetime.combine(day, dt.time.min, tzinfo=dt.timezone.utc).timestamp())


class _ResponseError(DbLayerError):
    """An error the fake is about to render as an HTTP response.

    Subclasses :class:`DbLayerError` so one ``except`` also catches what
    :func:`~prometheon.dbclient.auth.verify_request` raises, and carries the
    structured detail fields the contract defines rather than burying them in
    the message for something downstream to parse back out.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_code: DbErrorCode,
        detail: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message, status_code=status_code, error_code=error_code.value)
        self.detail: dict[str, int] = detail or {}


@dataclass
class DaySeed:
    """One day's data as the DB layer would have assembled it."""

    test_items: tuple[TestContentItem, ...] = ()
    production_items: tuple[ProductionContentItem, ...] = ()
    eligible_miners: tuple[EligibleMiner, ...] = ()
    model_commitments: tuple[ModelCommitment, ...] = ()
    #: ``None`` means the snapshot exists but has not been built yet.
    built_at: int | None = None


@dataclass
class _StoredEvaluation:
    submission: EvaluationSubmission
    accepted_at: int


@dataclass
class FakeDbLayer:
    """An in-memory implementation of the ``/v2`` contract."""

    netuid: int
    now: Callable[[], int] = _unix_now
    max_page_limit: int = DEFAULT_PAGE_LIMIT
    days: dict[dt.date, DaySeed] = field(default_factory=dict)
    evaluations: dict[tuple[dt.date, str], _StoredEvaluation] = field(default_factory=dict)
    #: ``(method, request target)`` for everything that arrived, authenticated
    #: or not. Lets a test assert what the client actually put on the wire.
    request_log: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.directory = StaticHotkeyDirectory({}, netuid=self.netuid)
        self.nonces = InMemoryNonceStore()

    # -- seeding -----------------------------------------------------------

    def register(self, hotkey: str, role: CallerRole) -> None:
        self.directory.set_role(hotkey, role)

    def deregister(self, hotkey: str) -> None:
        self.directory.set_role(hotkey, None)

    def seed_day(
        self,
        day: dt.date,
        *,
        test_items: Sequence[TestContentItem] = (),
        production_items: Sequence[ProductionContentItem] = (),
        eligible_miners: Sequence[EligibleMiner] = (),
        model_commitments: Sequence[ModelCommitment] = (),
        built: bool = True,
        built_at: int | None = None,
    ) -> DaySeed:
        if len(production_items) > PRODUCTION_ITEM_CAP:
            raise DbLayerError(
                f"a day carries at most {PRODUCTION_ITEM_CAP} production items, "
                f"got {len(production_items)}"
            )
        # A day's snapshot can only be complete once the day is over, so the
        # default build time is the following midnight rather than "now".
        default_built_at = _midnight_utc(day) + _SECONDS_PER_DAY
        seed = DaySeed(
            test_items=tuple(test_items),
            production_items=tuple(production_items),
            eligible_miners=tuple(eligible_miners),
            model_commitments=tuple(model_commitments),
            built_at=(built_at if built_at is not None else default_built_at) if built else None,
        )
        self.days[day] = seed
        return seed

    def mark_not_ready(self, day: dt.date) -> None:
        seed = self.days.get(day)
        if seed is None:
            raise DbLayerError(f"nothing seeded for {day.isoformat()}")
        seed.built_at = None

    def manifest(self, day: dt.date) -> SnapshotManifest:
        seed, built_at = self._require_built(day)
        return SnapshotManifest(
            date=day,
            snapshot_version=SNAPSHOT_VERSION,
            built_at=built_at,
            test_item_count=len(seed.test_items),
            production_item_count=len(seed.production_items),
            eligible_miner_count=len(seed.eligible_miners),
            model_commitment_count=len(seed.model_commitments),
            content_hash=snapshot_content_hash(
                date=day,
                test_items=seed.test_items,
                production_items=seed.production_items,
                eligible_miners=seed.eligible_miners,
                model_commitments=seed.model_commitments,
            ),
        )

    def stored_evaluations(self) -> tuple[EvaluationSubmission, ...]:
        return tuple(stored.submission for stored in self.evaluations.values())

    # -- transport ---------------------------------------------------------

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        """Serve one request, rendering any failure as the contract's error body."""
        target = request.url.raw_path.decode("ascii")
        self.request_log.append((request.method, target))
        try:
            return self._serve(request, target)
        except DbLayerError as exc:
            return _error_response(exc)

    def _serve(self, request: httpx.Request, target: str) -> httpx.Response:
        route = _route_of(target)
        if route is None:
            raise _ResponseError(
                f"no such endpoint: {target}",
                status_code=404,
                error_code=DbErrorCode.NOT_FOUND,
            )

        caller = verify_request(
            method=request.method,
            path=target,
            body=request.content,
            headers=dict(request.headers),
            netuid=self.netuid,
            directory=self.directory,
            nonce_store=self.nonces,
            now=self.now(),
        )

        if request.method != route.method:
            raise _ResponseError(
                f"{route.path} does not accept {request.method}",
                status_code=405,
                error_code=DbErrorCode.BAD_REQUEST,
            )

        if route.name == "evaluations":
            return self._post_evaluation(caller, request)

        day = _parse_day(route.day)
        if route.name == "dataset":
            # Content is the only thing under embargo; the manifest, the
            # eligible-miner list, and the commitment set carry none of it and
            # are validator-only for a different reason.
            self._check_embargo(caller, day)
        else:
            require_validator(caller)

        if route.name == "snapshot":
            return _json_response(200, self.manifest(day))
        if route.name == "eligible-miners":
            seed, _ = self._require_built(day)
            return _json_response(200, EligibleMinerSet(date=day, miners=seed.eligible_miners))
        if route.name == "models":
            seed, _ = self._require_built(day)
            return _json_response(
                200,
                ModelCommitmentSet(
                    date=day, frozen_at=_midnight_utc(day), commitments=seed.model_commitments
                ),
            )
        return self._get_dataset(request, day, route.collection)

    # -- endpoint bodies ---------------------------------------------------

    def _get_dataset(self, request: httpx.Request, day: dt.date, collection: str) -> httpx.Response:
        seed, _ = self._require_built(day)
        limit = self._page_limit(request)
        offset = _decode_cursor(request.url.params.get("cursor"), day, collection)

        if collection == "test":
            total = len(seed.test_items)
            return _json_response(
                200,
                TestContentPage(
                    date=day,
                    items=seed.test_items[offset : offset + limit],
                    next_cursor=_next_cursor(day, collection, offset + limit, total),
                    total_count=total,
                ),
            )

        total = len(seed.production_items)
        return _json_response(
            200,
            ProductionContentPage(
                date=day,
                items=seed.production_items[offset : offset + limit],
                next_cursor=_next_cursor(day, collection, offset + limit, total),
                total_count=total,
            ),
        )

    def _post_evaluation(
        self, caller: AuthenticatedCaller, request: httpx.Request
    ) -> httpx.Response:
        require_validator(caller)
        try:
            submission = EvaluationSubmission.model_validate_json(request.content)
        except ValidationError as exc:
            raise _ResponseError(
                f"evaluation payload is not acceptable: {exc}",
                status_code=422,
                error_code=DbErrorCode.INVALID_PAYLOAD,
            ) from exc

        if submission.validator_hotkey != caller.hotkey:
            raise _ResponseError(
                f"evaluation names {submission.validator_hotkey} but was sent by {caller.hotkey}",
                status_code=403,
                error_code=DbErrorCode.NOT_AUTHORIZED,
            )
        if submission.netuid != self.netuid:
            raise _ResponseError(
                f"evaluation is for netuid {submission.netuid}, this service serves {self.netuid}",
                status_code=400,
                error_code=DbErrorCode.BAD_REQUEST,
            )
        try:
            verify_evaluation(submission)
        except SignatureError as exc:
            raise _ResponseError(
                f"evaluation signature does not verify: {exc}",
                status_code=422,
                error_code=DbErrorCode.INVALID_PAYLOAD,
            ) from exc

        key = (submission.date, submission.validator_hotkey)
        existing = self.evaluations.get(key)
        if existing is not None:
            # Re-sending the identical record is how a client that lost the
            # response finds out it landed. Sending a *different* record for
            # the same day is a validator revising a published result, and that
            # must not be silently accepted.
            if existing.submission.signature != submission.signature:
                raise _ResponseError(
                    f"{submission.validator_hotkey} already published a different result for "
                    f"{submission.date.isoformat()}",
                    status_code=409,
                    error_code=DbErrorCode.DUPLICATE_EVALUATION,
                )
            return _json_response(
                200,
                EvaluationAccepted(
                    evaluation_id=_evaluation_id(key),
                    accepted_at=existing.accepted_at,
                    duplicate=True,
                ),
            )

        accepted_at = self.now()
        self.evaluations[key] = _StoredEvaluation(submission=submission, accepted_at=accepted_at)
        return _json_response(
            201,
            EvaluationAccepted(
                evaluation_id=_evaluation_id(key), accepted_at=accepted_at, duplicate=False
            ),
        )

    # -- helpers -----------------------------------------------------------

    def _require_built(self, day: dt.date) -> tuple[DaySeed, int]:
        seed = self.days.get(day)
        if seed is None:
            raise _ResponseError(
                f"no snapshot exists for {day.isoformat()}",
                status_code=404,
                error_code=DbErrorCode.NOT_FOUND,
            )
        if seed.built_at is None:
            raise _ResponseError(
                f"the snapshot for {day.isoformat()} has not been built yet",
                status_code=409,
                error_code=DbErrorCode.SNAPSHOT_NOT_READY,
                detail={"retry_after_seconds": _SNAPSHOT_RETRY_AFTER_SECONDS},
            )
        return seed, seed.built_at

    def _check_embargo(self, caller: AuthenticatedCaller, day: dt.date) -> None:
        if caller.role is CallerRole.VALIDATOR:
            return
        lifts_on = day + dt.timedelta(days=EMBARGO_DAYS)
        available_at = _midnight_utc(lifts_on)
        if self.now() < available_at:
            raise _ResponseError(
                f"content for {day.isoformat()} is embargoed from miners until "
                f"{lifts_on.isoformat()} 00:00 UTC",
                status_code=403,
                error_code=DbErrorCode.EMBARGOED,
                detail={"available_at": available_at},
            )

    def _page_limit(self, request: httpx.Request) -> int:
        raw = request.url.params.get("limit")
        if raw is None:
            return self.max_page_limit
        try:
            requested = int(raw)
        except ValueError as exc:
            raise _ResponseError(
                f"limit must be an integer, got {raw!r}",
                status_code=400,
                error_code=DbErrorCode.BAD_REQUEST,
            ) from exc
        if not 1 <= requested <= MAX_PAGE_LIMIT:
            raise _ResponseError(
                f"limit must be between 1 and {MAX_PAGE_LIMIT}, got {requested}",
                status_code=400,
                error_code=DbErrorCode.BAD_REQUEST,
            )
        return min(requested, self.max_page_limit)


@dataclass(frozen=True)
class _Route:
    name: str
    method: str
    path: str
    day: str = ""
    collection: str = ""


def _route_of(target: str) -> _Route | None:
    """Match a request target, ignoring any base path the deployment sits under.

    Routing on the ``/v2`` suffix rather than on the whole path means the fake
    works behind any base URL a test picks, while the *signed* path stays the
    full target, which is the property worth proving.
    """
    path = target.split("?", 1)[0]
    marker = path.find("/v2/")
    if marker < 0:
        return None
    tail = path[marker:]

    if tail == _EVALUATIONS_PATH:
        return _Route(name="evaluations", method="POST", path=tail)
    snapshot = _SNAPSHOT_RE.match(tail)
    if snapshot:
        return _Route(name="snapshot", method="GET", path=tail, day=snapshot.group("date"))
    dataset = _DATASET_RE.match(tail)
    if dataset:
        return _Route(
            name="dataset",
            method="GET",
            path=tail,
            day=dataset.group("date"),
            collection=dataset.group("collection"),
        )
    miners = _MINERS_RE.match(tail)
    if miners:
        return _Route(name="eligible-miners", method="GET", path=tail, day=miners.group("date"))
    models = _MODELS_RE.match(tail)
    if models:
        return _Route(name="models", method="GET", path=tail, day=models.group("date"))
    return None


def _parse_day(raw: str) -> dt.date:
    try:
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise _ResponseError(
            f"{raw!r} is not a calendar date",
            status_code=400,
            error_code=DbErrorCode.BAD_REQUEST,
        ) from exc


def _evaluation_id(key: tuple[dt.date, str]) -> str:
    day, hotkey = key
    return f"{day.isoformat()}:{hotkey}"


def _next_cursor(day: dt.date, collection: str, next_offset: int, total: int) -> str | None:
    return _encode_cursor(day, collection, next_offset) if next_offset < total else None


def _encode_cursor(day: dt.date, collection: str, offset: int) -> str:
    raw = json.dumps(
        {"c": collection, "d": day.isoformat(), "o": offset},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None, day: dt.date, collection: str) -> int:
    """Read an offset back out, refusing a cursor from anywhere else.

    A cursor is opaque to the client but not to the server: one minted for a
    different day or collection would silently paginate the wrong corpus, so it
    is rejected rather than coerced.
    """
    if cursor is None:
        return 0
    bad = _ResponseError(
        f"cursor is not valid for {collection} on {day.isoformat()}",
        status_code=400,
        error_code=DbErrorCode.BAD_CURSOR,
    )
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded: Any = json.loads(base64.urlsafe_b64decode(cursor + padding))
    except (ValueError, TypeError) as exc:
        raise bad from exc
    if not isinstance(decoded, dict):
        raise bad
    if decoded.get("d") != day.isoformat() or decoded.get("c") != collection:
        raise bad
    offset = decoded.get("o")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise bad
    return offset


def _json_response(status: int, model: BaseModel) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        content=model.model_dump_json().encode("utf-8"),
        headers={"content-type": "application/json"},
    )


def _error_response(exc: DbLayerError) -> httpx.Response:
    body: dict[str, Any] = {
        "code": exc.error_code or DbErrorCode.INTERNAL.value,
        "message": str(exc),
    }
    if isinstance(exc, _ResponseError):
        body.update(exc.detail)
    return httpx.Response(
        status_code=exc.status_code or 500,
        json={"error": body},
        headers={"content-type": "application/json"},
    )


__all__ = ["DaySeed", "FakeDbLayer"]
