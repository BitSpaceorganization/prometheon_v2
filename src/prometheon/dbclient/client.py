"""HTTP client for the subnet DB layer.

Synchronous, because a validator's ingest is one linear step at the top of the
cycle and there is nothing to overlap it with. The transport is injectable, so
every test in this repository runs the real client against the real fake over
``httpx.MockTransport``: no network, and no mock that only agrees with itself.

Two behaviours here are not obvious.

**Every retry attempt gets a fresh nonce and a fresh ``issued_at``.** Reusing
the envelope would make the second attempt a textbook replay, which the server
is required to reject, so a client that signs once and retries three times
fails every retry it makes. This is the kind of defect that only appears under
the load that made you want retries in the first place.

**Transport errors are not retried.** Only 429, 502, 503, and 504 are. Those
four are the server saying it did not process the request; a connection reset
says nothing, and a blind retry of ``POST /v2/evaluations`` over one can store
the cycle twice. The caller retries the step, where it can see whether that is
safe.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final, TypeVar

import httpx
from bittensor_wallet import Keypair
from pydantic import BaseModel, ValidationError

from prometheon.config import Config
from prometheon.dbclient import auth
from prometheon.dbclient.models import (
    API_VERSION,
    CURSOR_RE,
    MAX_PAGE_LIMIT,
    SNAPSHOT_VERSION,
    DbErrorCode,
    EligibleMiner,
    EligibleMinerSet,
    ErrorResponse,
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
from prometheon.errors import (
    DbLayerError,
    EmbargoedError,
    NotAuthorizedError,
    SnapshotNotReadyError,
)

_T = TypeVar("_T", bound=BaseModel)

#: The only statuses worth sending the same request again for.
RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({429, 502, 503, 504})

_BACKOFF_BASE_SECONDS: Final[float] = 0.5
_BACKOFF_CAP_SECONDS: Final[float] = 30.0

#: Longest ``Retry-After`` the client will honour. A server asking for more
#: than this has effectively told the cycle to fail, and waiting silently past
#: the daily window helps nobody.
_MAX_RETRY_AFTER_SECONDS: Final[float] = 60.0

#: Refuses to walk a paginated collection forever. A server that keeps handing
#: back pages is a bug the client should surface, not absorb.
_MAX_PAGES: Final[int] = 1000

_AUTH_CODES: Final[frozenset[str]] = frozenset(
    {
        DbErrorCode.AUTH_MALFORMED.value,
        DbErrorCode.AUTH_INVALID.value,
        DbErrorCode.AUTH_EXPIRED.value,
        DbErrorCode.AUTH_REPLAYED.value,
        DbErrorCode.NOT_AUTHORIZED.value,
    }
)


def _unix_now() -> int:
    return int(time.time())


@dataclass(frozen=True)
class DaySnapshot:
    """Everything a cycle needs about one day, already checked for agreement."""

    manifest: SnapshotManifest
    test_items: tuple[TestContentItem, ...]
    production_items: tuple[ProductionContentItem, ...]
    eligible_miners: tuple[EligibleMiner, ...]
    model_commitments: tuple[ModelCommitment, ...]
    #: Unix seconds at which the DB layer froze the commitment set for this day.
    #:
    #: Carried rather than discarded because it is the only evidence that the
    #: freeze a validator was promised actually happened. Dropping it made the
    #: field unassertable by any caller of this client, which is what it exists
    #: for.
    model_commitments_frozen_at: int = 0


class DbClient:
    """Talks to the subnet DB layer as one hotkey."""

    def __init__(
        self,
        *,
        base_url: str,
        netuid: int,
        keypair: Keypair,
        hotkey: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
        now: Callable[[], int] = _unix_now,
        nonce_factory: Callable[[], str] = auth.new_nonce,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_retries < 0:
            raise DbLayerError("max_retries must not be negative")
        self._netuid = netuid
        self._keypair = keypair
        self._hotkey = hotkey or str(keypair.ss58_address)
        self._max_retries = max_retries
        self._now = now
        self._nonce_factory = nonce_factory
        self._sleep = sleep
        # A trailing slash makes URL joining append rather than replace the
        # last path segment, so a base URL with a prefix keeps its prefix.
        normalised = base_url if base_url.endswith("/") else base_url + "/"
        self._base = httpx.URL(normalised)
        self._http = httpx.Client(
            transport=transport,
            timeout=httpx.Timeout(timeout_seconds),
            # A redirect changes the path, and the path is signed. Following
            # one would send a signature that cannot verify and produce a
            # confusing 401 instead of an honest error here.
            follow_redirects=False,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        keypair: Keypair,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> DbClient:
        return cls(
            base_url=config.db.base_url,
            netuid=config.chain.netuid,
            keypair=keypair,
            timeout_seconds=config.db.request_timeout_seconds,
            max_retries=config.db.max_retries,
            transport=transport,
        )

    @property
    def hotkey(self) -> str:
        return self._hotkey

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DbClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- endpoints ---------------------------------------------------------

    def get_snapshot(self, day: dt.date) -> SnapshotManifest:
        return self._get(f"{API_VERSION}/snapshot/{day.isoformat()}", model=SnapshotManifest)

    def get_test_page(
        self, day: dt.date, *, cursor: str | None = None, limit: int | None = None
    ) -> TestContentPage:
        path = f"{API_VERSION}/dataset/{day.isoformat()}/test"
        return self._get(_with_query(path, cursor, limit), model=TestContentPage)

    def get_production_page(
        self, day: dt.date, *, cursor: str | None = None, limit: int | None = None
    ) -> ProductionContentPage:
        path = f"{API_VERSION}/dataset/{day.isoformat()}/production"
        return self._get(_with_query(path, cursor, limit), model=ProductionContentPage)

    def iter_test_content(
        self, day: dt.date, *, limit: int | None = None
    ) -> Iterator[TestContentItem]:
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            page = self.get_test_page(day, cursor=cursor, limit=limit)
            _check_page(page.date, day, page.next_cursor, cursor, seen)
            if page.next_cursor is not None:
                seen.add(page.next_cursor)
            yield from page.items
            if page.next_cursor is None:
                return
            cursor = page.next_cursor
        raise DbLayerError(f"test content for {day.isoformat()} exceeded {_MAX_PAGES} pages")

    def iter_production_content(
        self, day: dt.date, *, limit: int | None = None
    ) -> Iterator[ProductionContentItem]:
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_PAGES):
            page = self.get_production_page(day, cursor=cursor, limit=limit)
            _check_page(page.date, day, page.next_cursor, cursor, seen)
            if page.next_cursor is not None:
                seen.add(page.next_cursor)
            yield from page.items
            if page.next_cursor is None:
                return
            cursor = page.next_cursor
        raise DbLayerError(f"production content for {day.isoformat()} exceeded {_MAX_PAGES} pages")

    def fetch_test_content(self, day: dt.date) -> tuple[TestContentItem, ...]:
        return tuple(self.iter_test_content(day))

    def fetch_production_content(self, day: dt.date) -> tuple[ProductionContentItem, ...]:
        return tuple(self.iter_production_content(day))

    def get_eligible_miners(self, day: dt.date) -> EligibleMinerSet:
        return self._get(f"{API_VERSION}/eligible-miners/{day.isoformat()}", model=EligibleMinerSet)

    def get_model_commitments(self, day: dt.date) -> ModelCommitmentSet:
        return self._get(f"{API_VERSION}/models/{day.isoformat()}", model=ModelCommitmentSet)

    def get_evaluation(self, day: dt.date, validator_hotkey: str) -> EvaluationSubmission:
        """One validator's published record for a day.

        The read side of :meth:`submit_evaluation`. Returned verbatim, signature
        included, because the caller has to verify it against the hotkey it
        names rather than against whoever served it -- this client speaks to the
        DB layer, and the DB layer is a service this code audits rather than
        believes.
        """
        return self._get(
            f"/v2/evaluations/{day.isoformat()}/{validator_hotkey}",
            model=EvaluationSubmission,
        )

    def submit_evaluation(self, submission: EvaluationSubmission) -> EvaluationAccepted:
        if not submission.signature:
            raise DbLayerError(
                "evaluation is unsigned; sign it with auth.sign_evaluation before submitting"
            )
        body = submission.model_dump_json(exclude_none=False).encode("utf-8")
        return self._request(
            "POST", f"{API_VERSION}/evaluations", model=EvaluationAccepted, body=body
        )

    def fetch_day(self, day: dt.date, *, verify_content_hash: bool = True) -> DaySnapshot:
        """Pull a whole day and check it against its own manifest.

        The recomputation is the reason this method exists. Every validator
        that fetches day N derives the same hash from what it was served, so a
        DB layer that serves two different corpora to two validators is caught
        by both of them at ingest, rather than showing up weeks later as an
        unexplained weight divergence.
        """
        manifest = self.get_snapshot(day)
        test_items = self.fetch_test_content(day)
        production_items = self.fetch_production_content(day)
        eligible = self.get_eligible_miners(day)
        models = self.get_model_commitments(day)

        # Every collection must agree about which day it is for. Only the
        # manifest date was checked, so a DB layer serving day N's manifest
        # alongside day N-1's miner set went unremarked. The content hash would
        # not catch it either, because it is computed over whatever was served.
        for name, served in (
            ("eligible miners", eligible.date),
            ("model commitments", models.date),
        ):
            if served != day:
                raise DbLayerError(
                    f"asked for {day.isoformat()} but the {name} collection is "
                    f"stamped {served.isoformat()}"
                )

        snapshot = DaySnapshot(
            manifest=manifest,
            test_items=test_items,
            production_items=production_items,
            eligible_miners=eligible.miners,
            model_commitments=models.commitments,
            model_commitments_frozen_at=models.frozen_at,
        )
        if verify_content_hash:
            _verify_snapshot(day, snapshot)
        return snapshot

    # -- transport ---------------------------------------------------------

    def _get(self, relative: str, *, model: type[_T]) -> _T:
        return self._request("GET", relative, model=model)

    def _request(
        self, method: str, relative: str, *, model: type[_T], body: bytes | None = None
    ) -> _T:
        url = self._base.join(relative)
        # The signed path is the origin-form request target exactly as it goes
        # on the wire, query string included, so the server can rebuild it from
        # what it received without either side normalising anything.
        signed_path = url.raw_path.decode("ascii")
        payload = body or b""

        attempt = 0
        while True:
            headers = auth.signed_headers(
                self._keypair,
                method=method,
                path=signed_path,
                body=payload,
                netuid=self._netuid,
                hotkey=self._hotkey,
                nonce=self._nonce_factory(),
                issued_at=self._now(),
            )
            if body is not None:
                headers["content-type"] = "application/json"

            try:
                response = self._http.request(method, url, headers=headers, content=payload)
            except httpx.HTTPError as exc:
                raise DbLayerError(f"{method} {signed_path} failed in transport: {exc}") from exc

            if response.status_code in RETRYABLE_STATUSES and attempt < self._max_retries:
                self._sleep(_backoff_seconds(attempt, response.headers.get("retry-after")))
                attempt += 1
                continue

            if response.status_code >= 400:
                raise _error_for(response, method, signed_path)

            if 300 <= response.status_code < 400:
                # Named rather than left to fail as a schema error, because the
                # fix is a deployment change and nothing about "field missing"
                # would point an operator at it.
                raise DbLayerError(
                    f"{method} {signed_path} was redirected to "
                    f"{response.headers.get('location', 'an unnamed location')}; redirects are "
                    "not followed because the signature covers the path",
                    status_code=response.status_code,
                )

            return _parse(response, model, method, signed_path)


def _with_query(path: str, cursor: str | None, limit: int | None) -> str:
    """Append the paging query, rejecting anything that would need escaping.

    Cursors are contractually base64url, so a cursor containing a character
    outside that alphabet is a server bug, and one that would silently change
    the signed path if it were percent-encoded on the way out.
    """
    parts: list[str] = []
    if cursor is not None:
        if not CURSOR_RE.match(cursor):
            raise DbLayerError(f"cursor is not base64url: {cursor!r}")
        parts.append(f"cursor={cursor}")
    if limit is not None:
        if not 1 <= limit <= MAX_PAGE_LIMIT:
            raise DbLayerError(f"limit must be between 1 and {MAX_PAGE_LIMIT}, got {limit}")
        parts.append(f"limit={limit}")
    return path if not parts else path + "?" + "&".join(parts)


def _check_page(
    page_day: dt.date,
    requested: dt.date,
    next_cursor: str | None,
    previous: str | None,
    seen: set[str] | None = None,
) -> None:
    """Reject a page that is for the wrong day or does not advance.

    ``seen`` holds every cursor already presented for this collection. Checking
    only against the immediately preceding cursor caught a self-loop but not a
    cycle of length two: a server alternating ``AAAA``/``BBBB`` advanced on
    every step, re-yielded the same items, and ran to the page cap. A thousand
    requests and a thousand duplicated items before anything complained.
    """
    if page_day != requested:
        raise DbLayerError(
            f"asked for {requested.isoformat()} and was served {page_day.isoformat()}"
        )
    if next_cursor is None:
        return
    if not CURSOR_RE.match(next_cursor):
        raise DbLayerError(f"next_cursor is not base64url: {next_cursor!r}")
    if next_cursor == previous:
        raise DbLayerError("server returned the cursor it was given; pagination is not advancing")
    if seen is not None and next_cursor in seen:
        raise DbLayerError(
            f"server re-issued cursor {next_cursor!r}, which it has already served; "
            "pagination is cycling rather than advancing"
        )


def _verify_snapshot(day: dt.date, snapshot: DaySnapshot) -> None:
    manifest = snapshot.manifest
    # Checked before the hash is recomputed. The preimage folds in this
    # client's own constant, so a served version this build does not implement
    # would surface as "content hash mismatch", pointing an operator at data
    # corruption when the real cause is a version skew needing a client
    # upgrade.
    if manifest.snapshot_version != SNAPSHOT_VERSION:
        raise DbLayerError(
            f"snapshot for {day.isoformat()} declares version "
            f"{manifest.snapshot_version!r}, but this build implements "
            f"{SNAPSHOT_VERSION!r}; upgrade the validator rather than reading it"
        )
    if manifest.date != day:
        raise DbLayerError(
            f"manifest is for {manifest.date.isoformat()}, asked for {day.isoformat()}"
        )
    counted: Sequence[tuple[str, int, int]] = (
        ("test_item_count", manifest.test_item_count, len(snapshot.test_items)),
        (
            "production_item_count",
            manifest.production_item_count,
            len(snapshot.production_items),
        ),
        ("eligible_miner_count", manifest.eligible_miner_count, len(snapshot.eligible_miners)),
        (
            "model_commitment_count",
            manifest.model_commitment_count,
            len(snapshot.model_commitments),
        ),
    )
    for name, declared, actual in counted:
        if declared != actual:
            raise DbLayerError(
                f"manifest declared {name}={declared} but {actual} items were served"
            )

    recomputed = snapshot_content_hash(
        date=day,
        test_items=snapshot.test_items,
        production_items=snapshot.production_items,
        eligible_miners=snapshot.eligible_miners,
        model_commitments=snapshot.model_commitments,
    )
    if recomputed != manifest.content_hash:
        raise DbLayerError(
            f"content hash mismatch for {day.isoformat()}: manifest says "
            f"{manifest.content_hash}, the served data hashes to {recomputed}"
        )


def _backoff_seconds(attempt: int, retry_after: str | None) -> float:
    requested = _retry_after_seconds(retry_after)
    if requested is not None:
        return requested
    return min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2.0**attempt))


def _retry_after_seconds(header: str | None) -> float | None:
    """Honour a numeric ``Retry-After`` that sits inside our own ceiling.

    The HTTP-date form is ignored rather than parsed: reading it correctly
    needs the server's clock, and the server's clock is exactly what is not
    trustworthy when it is busy enough to be shedding requests.
    """
    if not header:
        return None
    try:
        seconds = int(header)
    except ValueError:
        return None
    # A zero is treated as absent rather than as "no wait". Proxies and load
    # shedders do emit `Retry-After: 0`, and honouring it literally turned the
    # one status that means *back off* into a tight retry loop against a server
    # that had just said it was shedding load.
    if 0 < seconds <= _MAX_RETRY_AFTER_SECONDS:
        return float(seconds)
    return None


def _parse(response: httpx.Response, model: type[_T], method: str, path: str) -> _T:
    try:
        raw: Any = response.json()
    except ValueError as exc:
        raise DbLayerError(
            f"{method} {path} returned a body that is not JSON",
            status_code=response.status_code,
        ) from exc
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise DbLayerError(
            f"{method} {path} returned a body this client cannot accept: {exc}",
            status_code=response.status_code,
        ) from exc


def _error_for(response: httpx.Response, method: str, path: str) -> DbLayerError:
    """Turn a non-2xx response into the typed error its code names.

    Mapping is by ``error.code`` first and status second: the code is the
    stable contract, and a status alone cannot separate "you are not a
    validator" from "come back in two days". Both are 403, and they need
    opposite responses.
    """
    status = response.status_code
    code: str | None = None
    message = f"{method} {path} failed with HTTP {status}"
    detail_suffix = ""

    try:
        parsed = ErrorResponse.model_validate(response.json())
    except (ValueError, ValidationError):
        parsed = None
    if parsed is not None:
        code = parsed.error.code
        message = f"{method} {path} failed with HTTP {status} ({code}): {parsed.error.message}"
        if parsed.error.retry_after_seconds is not None:
            detail_suffix = f" retry after {parsed.error.retry_after_seconds}s"
        elif parsed.error.available_at is not None:
            detail_suffix = f" available at unix {parsed.error.available_at}"

    message += detail_suffix
    if code == DbErrorCode.SNAPSHOT_NOT_READY.value:
        return SnapshotNotReadyError(message, status_code=status, error_code=code)
    if code == DbErrorCode.EMBARGOED.value:
        return EmbargoedError(message, status_code=status, error_code=code)
    if code in _AUTH_CODES or (code is None and status in (401, 403)):
        return NotAuthorizedError(message, status_code=status, error_code=code)
    return DbLayerError(message, status_code=status, error_code=code)


__all__ = ["RETRYABLE_STATUSES", "DaySnapshot", "DbClient"]
