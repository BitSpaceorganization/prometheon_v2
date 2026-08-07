"""Shared HTTP plumbing for the registry's independent verification calls.

The registry is the one place in the validator that must not trust anything it
is told: every fact about a miner's model is re-fetched from Hugging Face and
Chutes. That makes the transport itself part of the trust boundary, so it lives
here once rather than being re-implemented per provider, where the two copies
would drift.

Three properties carry that:

- Retries are bounded. A transient 503 from a provider must not cost a miner a
  day of emission, but an unbounded retry loop would let one unreachable host
  stall the whole cycle. Retries apply only to statuses that mean "try again".
  A 404 is a verdict.
- Every body has a byte ceiling. Repo and chute contents are miner-supplied,
  and reading an unbounded response into memory is a denial of service any
  miner could trigger, so bodies stream and stop at a cap.
- The transport is injectable. Tests drive the real request/response envelope
  through ``httpx.MockTransport`` instead of monkeypatching methods, so a
  change to a URL or a header is visible to the test that asserts it.

Synchronous. Verification runs once per cycle over a handful of requests per
miner; an async client would buy nothing here and would force an event loop on
every caller.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from types import TracebackType
from typing import Any, Final, TypeVar

import httpx

from prometheon.errors import RegistryError

#: Default ceiling for a JSON body. Generous for a revision listing or a chute
#: info blob, and far below anything that would threaten a validator's memory.
#: A repo whose file listing exceeds this is refused before the manifest check
#: would have refused it for having too many files anyway.
MAX_JSON_BYTES: Final[int] = 4 << 20

_T = TypeVar("_T")
_ClientT = TypeVar("_ClientT", bound="RegistryHttpClient")

#: Statuses that mean "the answer is not ready", never "the answer is no".
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_RETRIES: Final[int] = 2
DEFAULT_BACKOFF_SECONDS: Final[float] = 1.0


class _RetryableStatusError(Exception):
    """Internal signal: this response is worth another attempt."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _raise_for_status(response: httpx.Response, what: str) -> None:
    """Turn a response status into either a retry signal or a typed failure."""
    code = response.status_code
    if code in _RETRYABLE_STATUS:
        raise _RetryableStatusError(code)
    if code == 404:
        raise RegistryError(f"{what} does not exist (HTTP 404)")
    if code in (401, 403):
        raise RegistryError(f"{what} is not publicly readable (HTTP {code})")
    if code >= 400:
        raise RegistryError(f"{what} failed with HTTP {code}")


class RegistryHttpClient:
    """A small synchronous client that returns JSON objects or capped bytes."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 0:
            raise RegistryError("retries must not be negative")
        if timeout_seconds <= 0:
            raise RegistryError("timeout_seconds must be positive")
        self._retries = retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
            headers=dict(headers or {}),
            # Hugging Face answers a file fetch with a redirect to its CDN, so a
            # client that does not follow redirects sees an empty body and
            # concludes the engine hash is wrong.
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self: _ClientT) -> _ClientT:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- requests ---------------------------------------------------------

    def get_json(self, url: str, *, what: str, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
        """GET ``url`` and return a JSON object, or raise :class:`RegistryError`.

        Capped and streamed like :meth:`get_bytes`, because these bodies are
        miner-influenced too. A revision listing returns one ``siblings`` entry
        per file in the repository, and the file count is entirely the miner's
        choice. An unbounded ``.get()`` buffered the whole response and
        ``.json()`` materialised it, before any manifest check could reject the
        repo for having too many files in the first place.
        """

        def call() -> dict[str, Any]:
            raw = self._stream_capped(url, what=what, max_bytes=max_bytes)
            try:
                payload: Any = json.loads(raw)
            except ValueError as exc:
                raise RegistryError(f"{what} did not return JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise RegistryError(
                    f"{what} returned a JSON {type(payload).__name__}, expected an object"
                )
            return payload

        return self._with_retries(what, call)

    def get_bytes(self, url: str, *, what: str, max_bytes: int) -> bytes:
        """GET ``url`` and return at most ``max_bytes``, refusing anything larger."""
        return self._with_retries(
            what, lambda: self._stream_capped(url, what=what, max_bytes=max_bytes)
        )

    def _stream_capped(self, url: str, *, what: str, max_bytes: int) -> bytes:
        """Read a response body, stopping at ``max_bytes`` rather than buffering it.

        The cap is enforced while reading, not after: checking ``Content-Length``
        would trust a header, and checking the finished body would mean the
        memory has already been spent.
        """
        with self._client.stream("GET", url) as response:
            _raise_for_status(response, what)
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise RegistryError(f"{what} is larger than the {max_bytes} byte ceiling")
                chunks.append(chunk)
        return b"".join(chunks)

    def _with_retries(self, what: str, call: Callable[[], _T]) -> _T:
        last_error = "no attempt was made"
        for attempt in range(self._retries + 1):
            try:
                return call()
            except _RetryableStatusError as exc:
                last_error = str(exc)
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            if attempt < self._retries:
                self._sleep(self._backoff_seconds * (2**attempt))
        raise RegistryError(
            f"{what} is unavailable after {self._retries + 1} attempts ({last_error})"
        )


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_SECONDS",
    "RegistryHttpClient",
]
