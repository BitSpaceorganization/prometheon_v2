"""The chat-completions transport used for ground-truth labelling.

One method, one request shape, no provider SDK. The subnet needs an
OpenAI-compatible ``POST /chat/completions`` and nothing else, and a narrow
surface is what makes the transport injectable: every test in this package
drives the real client over an :class:`httpx.MockTransport`, so the envelope,
headers, and retry policy that ship are the ones under test.

Two failures are kept apart, because the batcher must react to them
differently:

- :class:`LabellingError` means the endpoint is unusable: no key, rejected
  auth, unreachable after retries. Nothing about the corpus will fix it, so it
  aborts the cycle.
- :class:`LabelResponseError` means the endpoint answered but the answer is not
  a usable one. That is the batcher's cue to split and retry, which is also the
  right response to a truncated completion.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol

import httpx

from prometheon.config import LabellingConfig
from prometheon.errors import LabellingError, LabelResponseError
from prometheon.version import __version__

_COMPLETIONS_PATH: Final[str] = "/chat/completions"

#: Retried: rate limiting, request timeouts, and anything the server calls its
#: own fault. Every other 4xx is a bug in this client or in the operator's
#: configuration, and retrying it only delays the error they need to see.
_RETRYABLE_STATUSES: Final[frozenset[int]] = frozenset({408, 409, 425, 429})

#: Success ends at 300, not at 400.
#:
#: ``httpx.Client`` does not follow redirects by default, so a 3xx arrives here
#: as a real response with an empty body. Treating it as success handed that
#: empty body to the completion parser, which reported "response body is not
#: JSON". That points an operator at the model when the actual cause is a
#: ``base_url`` that redirects (``http://`` to ``https://``, or a provider that
#: moved the path). Falling into the error branch instead names the status and
#: the ``Location``.
_REDIRECT_FLOOR: Final[int] = 300
_CLIENT_ERROR_FLOOR: Final[int] = 400
_SERVER_ERROR_FLOOR: Final[int] = 500

_BACKOFF_BASE_SECONDS: Final[float] = 1.0
_BACKOFF_CAP_SECONDS: Final[float] = 30.0

#: A ``Retry-After`` is honoured but never trusted with the whole cycle. An
#: hour-long value from a misconfigured gateway would stall the day's run.
_MAX_RETRY_AFTER_SECONDS: Final[float] = 60.0

#: How much of an error body is quoted back to the operator.
_ERROR_SNIPPET_CHARS: Final[int] = 300


@dataclass(frozen=True)
class ChatResponse:
    """One completion: the assistant text plus what it cost."""

    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatCompletionClient(Protocol):
    """What :mod:`prometheon.labelling.batch` needs from a labeller."""

    def complete(self, *, system_prompt: str, user_prompt: str) -> ChatResponse:
        """Return the model's reply, or raise ``LabelResponseError``/``LabellingError``."""
        ...


class OpenAICompatibleClient:
    """A chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: LabellingConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        json_response_format: bool = True,
    ) -> None:
        """Build a client, reading the key from the env var ``config`` names.

        The key is read here rather than per request so a missing credential
        fails before the cycle starts, not four hours in. ``json_response_format``
        is an escape hatch: most compatible servers honour the field, and a few
        older ones reject unknown request fields outright.
        """
        api_key = os.environ.get(config.api_key_env, "").strip()
        if not api_key:
            raise LabellingError(
                f"no labelling API key: set the {config.api_key_env} environment variable "
                "(named by labelling.api_key_env in the config)"
            )

        self._config = config
        self._sleep = sleep
        self._json_response_format = json_response_format
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout_seconds),
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"prometheon/{__version__}",
            },
        )

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def complete(self, *, system_prompt: str, user_prompt: str) -> ChatResponse:
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # Ground truth must not wander between validators or between the
            # first attempt and its retry, and sampling is the largest avoidable
            # source of that drift.
            "temperature": 0,
        }
        if self._json_response_format:
            body["response_format"] = {"type": "json_object"}

        return _parse_completion(self._post_with_retries(body))

    def _post_with_retries(self, body: dict[str, Any]) -> httpx.Response:
        attempts = self._config.max_retries + 1
        last_failure = "no attempt was made"
        cause: Exception | None = None

        for attempt in range(attempts):
            delay: float | None = None
            try:
                response = self._client.post(_COMPLETIONS_PATH, json=body)
            except httpx.TimeoutException as exc:
                cause = exc
                last_failure = f"request timed out after {self._config.request_timeout_seconds}s"
            except httpx.TransportError as exc:
                cause = exc
                last_failure = f"transport failure: {exc}"
            except httpx.HTTPError as exc:
                # Malformed URL and friends: configuration, not weather.
                raise LabellingError(f"labelling request could not be sent: {exc}") from exc
            else:
                if response.status_code < _REDIRECT_FLOOR:
                    return response
                if response.status_code < _CLIENT_ERROR_FLOOR:
                    raise LabellingError(
                        f"labelling endpoint redirected with HTTP {response.status_code} "
                        f"to {response.headers.get('Location', '<no Location header>')!r}. "
                        "Redirects are not followed, because a redirected "
                        "completion request is a misconfigured base_url rather "
                        "than a working endpoint — check [labelling] base_url"
                    )
                if not _is_retryable_status(response.status_code):
                    raise LabellingError(
                        f"labelling endpoint rejected the request with HTTP "
                        f"{response.status_code}: {_snippet(response)}"
                    )
                last_failure = f"HTTP {response.status_code}: {_snippet(response)}"
                delay = _retry_after_seconds(response)

            if attempt == attempts - 1:
                break
            self._sleep(_backoff_seconds(attempt) if delay is None else delay)

        raise LabellingError(
            f"labelling endpoint {self._config.base_url} failed "
            f"after {attempts} attempt(s): {last_failure}"
        ) from cause


def _is_retryable_status(status_code: int) -> bool:
    return status_code in _RETRYABLE_STATUSES or status_code >= _SERVER_ERROR_FLOOR


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped.

    No jitter: one validator holds one API credential, so there is no herd to
    spread, and a deterministic delay is one less thing to reason about when
    reading a stalled cycle's logs.
    """
    return min(_BACKOFF_BASE_SECONDS * (2.0**attempt), _BACKOFF_CAP_SECONDS)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """The server's own wait, when it gave a usable one in seconds.

    The header's ``delta-seconds`` form is digits and nothing else, so that is
    what is accepted. Parsing it with ``float`` was too permissive in a way
    that crashed the cycle: ``float("nan")`` succeeds, and ``NaN`` slips through
    every comparison guard (``nan < 0`` is ``False``, ``min(nan, 60.0)`` returns
    ``nan``), so it reached ``time.sleep``, which raises a bare ``ValueError``.
    One header from a broken or hostile gateway was enough to put an untyped
    exception in front of the caller.

    ``Retry-After: 0`` is treated as absent rather than as "no wait". Some
    proxies and load shedders emit it, and honouring it literally turns the one
    status that means *back off* into a tight retry loop against a server that
    has just said it is shedding load.
    """
    raw = response.headers.get("Retry-After", "").strip()
    if not raw.isdigit():
        # Covers the empty header, a negative value, NaN and infinity, and the
        # HTTP-date form, which is legal but rare here and would need a
        # trustworthy clock-skew estimate to parse. Fall back to our own
        # backoff.
        return None
    seconds = float(raw)
    if seconds <= 0:
        return None
    return min(seconds, _MAX_RETRY_AFTER_SECONDS)


def _snippet(response: httpx.Response) -> str:
    try:
        text = response.text
    except (UnicodeDecodeError, httpx.HTTPError):  # pragma: no cover - defensive
        return "<unreadable body>"
    return text[:_ERROR_SNIPPET_CHARS].strip() or "<empty body>"


def _parse_completion(response: httpx.Response) -> ChatResponse:
    """Pull the assistant text out of an OpenAI-shaped envelope."""
    try:
        payload: object = response.json()
    except ValueError as exc:
        raise LabelResponseError(f"labelling response body is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelResponseError("labelling response body is not a JSON object")

    choices: object = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LabelResponseError("labelling response contained no choices")

    choice: object = choices[0]
    if not isinstance(choice, dict):
        raise LabelResponseError("labelling response choice is not an object")

    # Truncation gets its own error: it is the one malformed response that a
    # smaller batch reliably fixes, and that is exactly what the caller does
    # next.
    if choice.get("finish_reason") == "length":
        raise LabelResponseError("labelling response was truncated by the model's output limit")

    message: object = choice.get("message")
    content: object = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LabelResponseError(
            "labelling response carried no assistant content (a refusal or an empty completion)"
        )

    usage: object = payload.get("usage")
    usage_map = usage if isinstance(usage, dict) else {}
    return ChatResponse(
        text=content,
        prompt_tokens=_token_count(usage_map, "prompt_tokens"),
        completion_tokens=_token_count(usage_map, "completion_tokens"),
    )


def _token_count(usage: dict[Any, Any], key: str) -> int:
    """A usage counter, or zero.

    Usage is telemetry, not ground truth, so a provider that omits or mangles
    it must not fail a batch that carried perfectly good verdicts.
    """
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


__all__ = [
    "ChatCompletionClient",
    "ChatResponse",
    "OpenAICompatibleClient",
]
