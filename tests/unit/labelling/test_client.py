"""The transport: what goes on the wire, and what a bad answer costs."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from prometheon.config import LabellingConfig
from prometheon.errors import LabellingError, LabelResponseError
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.version import __version__

from .conftest import Harness, completion_envelope

pytestmark = pytest.mark.unit

ANSWER = '{"labels": [{"id": "a", "violates": true}]}'


def _complete(harness: Harness) -> Any:
    return harness.client.complete(system_prompt="policy here", user_prompt="items here")


def test_the_request_is_an_openai_chat_completion(
    build_labeller: Callable[..., Harness],
) -> None:
    harness = build_labeller(lambda items: ANSWER)

    _complete(harness)

    request = harness.requests[0]
    body = json.loads(request.content)
    assert request.method == "POST"
    assert request.url.path == "/v1/chat/completions"
    assert body["model"] == LabellingConfig().model
    assert body["messages"] == [
        {"role": "system", "content": "policy here"},
        {"role": "user", "content": "items here"},
    ]
    # Ground truth must not move between a first attempt and its retry.
    assert body["temperature"] == LabellingConfig().temperature
    assert body["response_format"] == {"type": "json_object"}


def test_the_bearer_token_comes_from_the_env_var_the_config_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOUSE_LABELLER_KEY", "sk-from-the-named-var")
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=completion_envelope(ANSWER))

    config = LabellingConfig(api_key_env="HOUSE_LABELLER_KEY")
    with OpenAICompatibleClient(config, transport=httpx.MockTransport(respond)) as client:
        client.complete(system_prompt="s", user_prompt="u")

    assert seen[0].headers["authorization"] == "Bearer sk-from-the-named-var"
    assert seen[0].headers["content-type"] == "application/json"
    assert seen[0].headers["user-agent"] == f"prometheon/{__version__}"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_credential_fails_before_the_cycle_starts(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    """Discovering this four hours into a cycle is the failure being avoided."""
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)

    with pytest.raises(LabellingError, match="OPENAI_API_KEY"):
        OpenAICompatibleClient(LabellingConfig())


def test_the_json_response_format_can_be_turned_off_for_older_servers(
    build_labeller: Callable[..., Harness],
) -> None:
    harness = build_labeller(lambda items: ANSWER, json_response_format=False)

    _complete(harness)

    assert "response_format" not in json.loads(harness.requests[0].content)


def test_usage_is_reported_back(build_labeller: Callable[..., Harness]) -> None:
    harness = build_labeller(
        lambda items: httpx.Response(
            200, json=completion_envelope(ANSWER, prompt_tokens=4321, completion_tokens=99)
        )
    )

    response = _complete(harness)

    assert response.text == ANSWER
    assert (response.prompt_tokens, response.completion_tokens) == (4321, 99)


@pytest.mark.parametrize(
    "usage",
    [{}, {"prompt_tokens": "many"}, {"prompt_tokens": -5}, {"prompt_tokens": True}],
)
def test_unusable_usage_counts_as_zero_rather_than_failing_the_batch(
    build_labeller: Callable[..., Harness], usage: dict[str, Any]
) -> None:
    """Usage is telemetry; it must never cost a batch its verdicts."""
    envelope = completion_envelope(ANSWER)
    envelope["usage"] = usage

    harness = build_labeller(lambda items: httpx.Response(200, json=envelope))
    response = _complete(harness)

    assert response.prompt_tokens == 0
    assert response.text == ANSWER


# --- retries ----------------------------------------------------------------


def test_a_rate_limited_request_is_retried(build_labeller: Callable[..., Harness]) -> None:
    replies = [httpx.Response(429, json={"error": "slow down"})]

    def respond(items: list[dict[str, str]]) -> Any:
        return replies.pop(0) if replies else ANSWER

    harness = build_labeller(respond)
    response = _complete(harness)

    assert response.text == ANSWER
    assert harness.calls == 2
    assert harness.sleeps == [1.0]


def test_backoff_doubles_between_attempts(build_labeller: Callable[..., Harness]) -> None:
    harness = build_labeller(
        lambda items: httpx.Response(503, text="upstream unavailable"),
        config=LabellingConfig(max_retries=3),
    )

    with pytest.raises(LabellingError, match="4 attempt"):
        _complete(harness)

    assert harness.calls == 4
    assert harness.sleeps == [1.0, 2.0, 4.0]


def test_a_retry_after_header_is_honoured(build_labeller: Callable[..., Harness]) -> None:
    replies = [httpx.Response(429, headers={"Retry-After": "7"}, text="")]

    harness = build_labeller(lambda items: replies.pop(0) if replies else ANSWER)
    _complete(harness)

    assert harness.sleeps == [7.0]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("9999", 60.0),  # capped: a bad gateway must not stall the whole cycle
        ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0),  # date form: fall back to backoff
        ("-3", 1.0),
        ("", 1.0),
    ],
)
def test_an_unhelpful_retry_after_falls_back_to_our_own_backoff(
    build_labeller: Callable[..., Harness], header: str, expected: float
) -> None:
    replies = [httpx.Response(429, headers={"Retry-After": header} if header else {}, text="")]

    harness = build_labeller(lambda items: replies.pop(0) if replies else ANSWER)
    _complete(harness)

    assert harness.sleeps == [expected]


def test_a_rejected_request_is_not_retried(build_labeller: Callable[..., Harness]) -> None:
    """A bad key does not become good on the third try; show the operator now."""
    harness = build_labeller(lambda items: httpx.Response(401, text="invalid api key"))

    with pytest.raises(LabellingError, match="401"):
        _complete(harness)

    assert harness.calls == 1
    assert harness.sleeps == []


def test_the_rejection_message_quotes_the_endpoint_body(
    build_labeller: Callable[..., Harness],
) -> None:
    harness = build_labeller(lambda items: httpx.Response(403, text="quota exhausted"))

    with pytest.raises(LabellingError, match="quota exhausted"):
        _complete(harness)


def test_a_timeout_is_retried_then_reported(build_labeller: Callable[..., Harness]) -> None:
    def times_out(items: list[dict[str, str]]) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    harness = build_labeller(times_out, config=LabellingConfig(max_retries=1))

    with pytest.raises(LabellingError, match="timed out"):
        _complete(harness)

    assert harness.calls == 2


def test_a_transport_failure_is_retried_then_reported(
    build_labeller: Callable[..., Harness],
) -> None:
    def refuses(items: list[dict[str, str]]) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    harness = build_labeller(refuses, config=LabellingConfig(max_retries=0))

    with pytest.raises(LabellingError, match="connection refused"):
        _complete(harness)

    assert harness.calls == 1
    assert harness.sleeps == []


# --- unusable answers -------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"choices": []}, "no choices"),
        ({"nochoices": 1}, "no choices"),
        ({"choices": ["text"]}, "not an object"),
        ({"choices": [{"message": {"role": "assistant"}}]}, "no assistant content"),
        ({"choices": [{"message": {"content": None}}]}, "no assistant content"),
        ({"choices": [{"message": {"content": "   "}}]}, "no assistant content"),
        ([], "not a JSON object"),
    ],
)
def test_an_envelope_without_an_answer_is_a_response_error(
    build_labeller: Callable[..., Harness], payload: Any, reason: str
) -> None:
    """A response error, not a transport error: the caller splits and retries."""
    harness = build_labeller(lambda items: httpx.Response(200, json=payload))

    with pytest.raises(LabelResponseError, match=reason):
        _complete(harness)


def test_a_non_json_body_is_a_response_error(build_labeller: Callable[..., Harness]) -> None:
    harness = build_labeller(lambda items: httpx.Response(200, text="<html>gateway</html>"))

    with pytest.raises(LabelResponseError, match="not JSON"):
        _complete(harness)


def test_a_truncated_completion_is_named_as_such(
    build_labeller: Callable[..., Harness],
) -> None:
    """Truncation is the one bad answer a smaller batch reliably fixes."""
    harness = build_labeller(
        lambda items: httpx.Response(
            200, json=completion_envelope('{"labels": [{"id"', finish_reason="length")
        )
    )

    with pytest.raises(LabelResponseError, match="truncated"):
        _complete(harness)


def test_a_refusal_is_a_response_error_rather_than_a_crash(
    build_labeller: Callable[..., Harness],
) -> None:
    """Moderation corpora get refused by the labeller's own safety layer."""
    envelope = completion_envelope("")
    envelope["choices"][0]["message"] = {"role": "assistant", "content": None, "refusal": "no"}

    harness = build_labeller(lambda items: httpx.Response(200, json=envelope))

    with pytest.raises(LabelResponseError, match="refusal"):
        _complete(harness)


class TestAGatewayCannotCrashOrSpinTheCycle:
    """Every one of these is a header a broken proxy really does emit."""

    def test_a_nan_retry_after_does_not_reach_time_sleep(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        """`float("nan")` parses, then survives every comparison guard.

        `nan < 0` is False and `min(nan, 60.0)` returns nan, so it reached
        `time.sleep` and raised a bare `ValueError`, an untyped exception in
        front of a caller that is promised typed ones.
        """
        harness = build_labeller(
            lambda _items: httpx.Response(429, headers={"Retry-After": "nan"}, text="slow down")
        )
        with pytest.raises(LabellingError):
            harness.client.complete(system_prompt="s", user_prompt="u")

        assert all(delay == delay for delay in harness.sleeps), "a NaN reached the sleep function"
        assert all(delay >= 0 for delay in harness.sleeps)

    def test_a_zero_retry_after_falls_back_to_exponential_backoff(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        """Honouring it literally turns "back off" into a tight retry loop."""
        harness = build_labeller(
            lambda _items: httpx.Response(429, headers={"Retry-After": "0"}, text="shedding")
        )
        with pytest.raises(LabellingError):
            harness.client.complete(system_prompt="s", user_prompt="u")

        assert harness.sleeps
        assert all(delay > 0 for delay in harness.sleeps)

    def test_a_redirect_names_the_misconfiguration_not_the_model(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        """A 3xx is an empty body; parsing it blamed the model for a config error."""
        harness = build_labeller(
            lambda _items: httpx.Response(302, headers={"Location": "https://elsewhere/v1"})
        )
        with pytest.raises(LabellingError, match="redirect"):
            harness.client.complete(system_prompt="s", user_prompt="u")
