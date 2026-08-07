"""Calling one miner's deployed model over the corpus.

The failure semantics are blunt:

- **Transport error**: one retry, then the batch is scored incorrect. A dropped
  connection is worth exactly one more attempt; a model that needs several is
  not serving.
- **Timeout**: no retry. The read timeout is the serving budget, and retrying
  it only spends the cycle's clock re-confirming that the model is too slow.
- **Non-2xx, or a body that breaks the wrapper contract** (wrong verdict count,
  ids that do not echo, a non-boolean verdict, a missing usage block): scored
  incorrect immediately. There is no repair and no partial credit.

If a batch of 100 could return 60 usable verdicts, the cheapest strategy would
be to answer only the easy items and let the rest fail. So the unit of success
is the whole batch, and the unit of failure is too.

The client is injected, so tests drive every one of these paths through
``httpx.MockTransport`` over the real request and response objects rather than
through a stubbed client that agrees with itself.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from prometheon.errors import EvaluationError, ModelResponseError
from prometheon.evaluation.corpus import (
    MAX_ITEMS_PER_REQUEST,
    LabelledItem,
    iter_batches,
)
from prometheon.evaluation.result import MinerEvaluation, MinerEvaluationBuilder

#: Ceiling on a miner's response body.
#:
#: A conformant answer to 100 items is a few kilobytes; this is orders of
#: magnitude above that and still far below anything that threatens a
#: validator. It exists because the body is miner-controlled and httpx has no
#: default limit, exactly as `MAX_SOURCE_BYTES` exists in the registry.
MAX_RESPONSE_BYTES: Final[int] = 16 << 20

DEFAULT_BATCH_SIZE: Final[int] = MAX_ITEMS_PER_REQUEST

#: One retry, matching ``EvaluationConfig.transport_retries``.
DEFAULT_TRANSPORT_RETRIES: Final[int] = 1


@dataclass(frozen=True)
class MinerTarget:
    """Where one miner's verified deployment answers moderation requests."""

    hotkey: str
    chute_id: str
    moderate_url: str

    def __post_init__(self) -> None:
        for name, value in (
            ("hotkey", self.hotkey),
            ("chute_id", self.chute_id),
            ("moderate_url", self.moderate_url),
        ):
            if not value:
                raise EvaluationError(f"{name} must not be empty")
        # Parsed, not just prefix-checked. A string can start with "https://"
        # and still be something httpx cannot turn into a request. "https://"
        # alone raises `ValueError: unknown url type` from urllib deep inside
        # proxy resolution, and a hostname carrying a zero-width space raises
        # `httpx.InvalidURL`. Both escape mid-request, outside every `except`
        # clause here, which turns a caller mistake into an untyped exception
        # instead of the typed one this module promises.
        try:
            parsed = httpx.URL(self.moderate_url)
        except (httpx.InvalidURL, ValueError, UnicodeError) as exc:
            raise EvaluationError(
                f"moderate_url is not a usable URL: {self.moderate_url!r} ({exc})"
            ) from exc

        # The request carries the validator's own Chutes API key in an
        # Authorization header. Over plaintext that key is readable by anything
        # on the path, and one leaked validator key is every validator's
        # evaluation budget.
        if parsed.scheme != "https":
            raise EvaluationError(
                f"moderate_url must be https, got {self.moderate_url!r}: the request "
                "carries the validator's Chutes API key"
            )
        if not parsed.host:
            raise EvaluationError(f"moderate_url has no host: {self.moderate_url!r}")


def evaluate_miner(
    *,
    client: httpx.Client,
    target: MinerTarget,
    items: Sequence[LabelledItem],
    policy: str,
    policy_version: str,
    api_key: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    transport_retries: int = DEFAULT_TRANSPORT_RETRIES,
) -> MinerEvaluation:
    """Evaluate ``items`` against one miner's model and return the result.

    Never raises for a model-side failure. Those are scored, not propagated.
    :class:`~prometheon.errors.EvaluationError` is raised only for a caller
    mistake, before any request goes out.

    The request timeout belongs to ``client``, built from
    ``EvaluationConfig.request_timeout_seconds``, so one client can serve every
    miner in a cycle and every miner gets the same budget.
    """
    if not policy:
        raise EvaluationError("policy text must not be empty")
    if not policy_version:
        raise EvaluationError("policy_version must not be empty")
    if not api_key:
        raise EvaluationError(
            "a Chutes API key is required: without one every batch would fail and "
            "every miner would score zero"
        )
    if transport_retries < 0:
        raise EvaluationError(f"transport_retries must not be negative, got {transport_retries}")

    builder = MinerEvaluationBuilder(target.hotkey)
    attempts = 1 + transport_retries

    for batch in iter_batches(items, batch_size):
        try:
            predicted, total_tokens = _run_batch(
                client=client,
                target=target,
                batch=batch,
                policy=policy,
                policy_version=policy_version,
                api_key=api_key,
                attempts=attempts,
            )
        except ModelResponseError as exc:
            builder.record_failed_batch(batch, reason=f"{exc.code}: {exc}")
        else:
            builder.record_batch(batch, predicted, total_tokens=total_tokens)

    return builder.build()


def build_request_payload(
    *, policy: str, policy_version: str, batch: Sequence[LabelledItem]
) -> dict[str, Any]:
    """The exact body the canonical wrapper's ``/moderate`` expects."""
    return {
        "policy_version": policy_version,
        "policy": policy,
        "items": [{"id": item.item_id, "content": item.content} for item in batch],
    }


def _run_batch(
    *,
    client: httpx.Client,
    target: MinerTarget,
    batch: Sequence[LabelledItem],
    policy: str,
    policy_version: str,
    api_key: str,
    attempts: int,
) -> tuple[dict[str, bool], int]:
    status_code, raw = _post(
        client=client,
        target=target,
        payload=build_request_payload(policy=policy, policy_version=policy_version, batch=batch),
        api_key=api_key,
        attempts=attempts,
    )
    if status_code // 100 != 2:
        raise ModelResponseError(
            f"{target.chute_id} answered HTTP {status_code} for a batch of {len(batch)} items"
        )
    body = _decode_body(raw)
    return _parse_verdicts(body, batch), _parse_usage(body)


def _post(
    *,
    client: httpx.Client,
    target: MinerTarget,
    payload: dict[str, Any],
    api_key: str,
    attempts: int,
) -> tuple[int, bytes]:
    """One request, returning the status and a size-capped body.

    Streamed rather than buffered. The response is entirely miner-controlled
    and ``httpx`` imposes no default limit, so ``client.post`` would read a
    body of any size into the validator's memory before a single check ran.
    That is a denial of service any miner could trigger for free, and it is
    inconsistent with the registry, which already caps the equally
    miner-controlled wrapper source. The cap is enforced while reading, so an
    oversized body costs the budget and not the machine.
    """
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with client.stream(
                "POST",
                target.moderate_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept": "application/json",
                },
            ) as response:
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > MAX_RESPONSE_BYTES:
                        raise ModelResponseError(
                            f"{target.chute_id} returned more than "
                            f"{MAX_RESPONSE_BYTES} bytes for a batch of "
                            f"{len(payload.get('items', ()))} items"
                        )
                    chunks.append(chunk)
                return response.status_code, b"".join(chunks)
        except ModelResponseError:
            raise
        except httpx.TimeoutException as exc:
            # Checked before TransportError, which it subclasses: a timeout is
            # the serving budget being exceeded, not a connection blip, and
            # retrying it buys nothing but cycle time.
            raise ModelResponseError(
                f"{target.chute_id} did not answer within the request timeout: {exc}"
            ) from exc
        except httpx.TransportError as exc:
            # Genuinely transient: connection refused, reset, DNS. Worth a retry.
            last_error = exc
        except httpx.RequestError as exc:
            # Everything else httpx raises for a request, and it must be caught
            # here or it escapes `evaluate_miner` entirely and aborts the whole
            # cycle for every miner.
            #
            # `TransportError` is not the base class it looks like:
            # `DecodingError` and `TooManyRedirects` are `RequestError`
            # subclasses that are *not* `TransportError`. Since `client.post`
            # reads and content-decodes the body inline, a miner who returns
            # `Content-Encoding: gzip` over bytes that are not gzip raises
            # `DecodingError` straight out of the call. One header from one
            # miner would take down the 04:00 UTC cycle on every validator at
            # once, and surface as an untyped third-party exception.
            #
            # Not retried: a malformed body is a contract violation, and it
            # will be malformed the next time too. The batch is scored
            # incorrect and the cycle continues, which is how every other
            # model-side failure is handled here.
            raise ModelResponseError(
                f"{target.chute_id} returned a response that could not be read: {exc}"
            ) from exc
    raise ModelResponseError(
        f"{target.chute_id} was unreachable after {attempts} attempt(s): {last_error}"
    )


def _parse_verdicts(body: dict[str, Any], batch: Sequence[LabelledItem]) -> dict[str, bool]:
    verdicts = body.get("verdicts")
    if not isinstance(verdicts, list):
        raise ModelResponseError("response has no 'verdicts' list")
    if len(verdicts) != len(batch):
        raise ModelResponseError(f"expected {len(batch)} verdicts, got {len(verdicts)}")

    predicted: dict[str, bool] = {}
    for entry in verdicts:
        if not isinstance(entry, dict):
            raise ModelResponseError("a verdict is not an object")
        item_id = entry.get("id")
        violates = entry.get("violates")
        if not isinstance(item_id, str):
            raise ModelResponseError("a verdict carries no string id")
        # bool is a subclass of int, so this also rejects 0/1. An integer
        # verdict means the wrapper contract was not honoured, and guessing
        # what it meant is how a scoring bug becomes an emissions bug.
        if not isinstance(violates, bool):
            raise ModelResponseError(f"verdict for {item_id!r} is {violates!r}, not a boolean")
        if item_id in predicted:
            raise ModelResponseError(f"duplicate verdict for {item_id!r}")
        predicted[item_id] = violates

    expected_ids = {item.item_id for item in batch}
    if predicted.keys() != expected_ids:
        missing = sorted(expected_ids - predicted.keys())
        unknown = sorted(predicted.keys() - expected_ids)
        raise ModelResponseError(
            f"verdict ids do not match the batch (missing {missing[:3]}, unexpected {unknown[:3]})"
        )
    return predicted


def _parse_usage(body: dict[str, Any]) -> int:
    """Total tokens for the batch.

    A missing or malformed usage block fails the batch. Token cost discounts
    model score, so a deployment that omits it would be claiming free
    inference. That is the one failure a miner would choose for itself.
    """
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ModelResponseError("response has no 'usage' object")

    total = 0
    for field in ("prompt_tokens", "completion_tokens"):
        value = usage.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ModelResponseError(f"usage.{field} is {value!r}, not a token count")
        total += value
    return total


def _decode_body(raw: bytes) -> dict[str, Any]:
    try:
        body: Any = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModelResponseError(f"response body is not JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ModelResponseError(f"response body is {type(body).__name__}, not an object")
    return body


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_TRANSPORT_RETRIES",
    "MAX_RESPONSE_BYTES",
    "MinerTarget",
    "build_request_payload",
    "evaluate_miner",
]
