"""Batch evaluation against a miner's /moderate endpoint.

Every path is driven through ``httpx.MockTransport``. The request a test
asserts on is the request that would go on the wire, headers and body included,
rather than a stub agreeing with the code that called it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from prometheon.errors import EvaluationError, ModelResponseError
from prometheon.evaluation.corpus import ItemSource, LabelledItem
from prometheon.evaluation.result import MinerEvaluation
from prometheon.evaluation.runner import MAX_RESPONSE_BYTES, MinerTarget, evaluate_miner

pytestmark = pytest.mark.unit

POLICY = "the policy text"
POLICY_VERSION = "2026-08-07"
API_KEY = "cpk_validator_key"

TARGET = MinerTarget(
    hotkey="5FA0miner_a",
    chute_id="chute-abc",
    moderate_url="https://miner-a.chutes.ai/moderate",
)

Handler = Callable[[httpx.Request], httpx.Response]


class Calls:
    """Records every request the runner actually sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def __len__(self) -> int:
        return len(self.requests)


def items(count: int, *, violating_every: int = 2) -> list[LabelledItem]:
    """A production-only corpus slice, alternating labels by default."""
    return [
        LabelledItem(
            item_id=f"p{index:03d}",
            content=f"content {index}",
            violating=index % violating_every == 0,
            source=ItemSource.PRODUCTION,
        )
        for index in range(count)
    ]


def make_client(handler: Handler, calls: Calls) -> httpx.Client:
    def recording(request: httpx.Request) -> httpx.Response:
        calls.requests.append(request)
        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(recording))


def perfect(request: httpx.Request) -> httpx.Response:
    """A wrapper-conformant response that answers every item correctly."""
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "wrapper_version": "prometheon-moderation/1",
            "policy_version": body["policy_version"],
            "verdicts": [
                # The corpus alternates labels by index parity, so a model that
                # answers on parity is a perfect model for this fixture.
                {"id": item["id"], "violates": int(item["id"][1:]) % 2 == 0}
                for item in body["items"]
            ],
            "usage": {
                "prompt_tokens": 10 * len(body["items"]),
                "completion_tokens": len(body["items"]),
                "items": len(body["items"]),
            },
        },
    )


def run(
    handler: Handler,
    *,
    count: int = 250,
    calls: Calls | None = None,
    batch_size: int = 100,
    transport_retries: int = 1,
) -> MinerEvaluation:
    recorder = calls if calls is not None else Calls()
    with make_client(handler, recorder) as client:
        return evaluate_miner(
            client=client,
            target=TARGET,
            items=items(count),
            policy=POLICY,
            policy_version=POLICY_VERSION,
            api_key=API_KEY,
            batch_size=batch_size,
            transport_retries=transport_retries,
        )


def response_body(response: httpx.Response) -> dict[str, Any]:
    body: dict[str, Any] = response.json()
    return body


def failing_batch_handler(handler: Handler, *, fail_batch: int, batch_size: int = 100) -> Handler:
    """Wrap a handler so one batch is unreachable on every attempt.

    Keyed on the batch's contents rather than on a call counter, so the retry
    hits the same failure the first attempt did. A counter would let the retry
    land on the next batch's turn and quietly succeed.
    """

    def wrapped(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        first_index = int(body["items"][0]["id"][1:])
        if first_index // batch_size == fail_batch:
            raise httpx.ConnectError("connection refused")
        return handler(request)

    return wrapped


# --- the happy path ---------------------------------------------------------


def test_a_healthy_model_is_scored_over_every_batch() -> None:
    calls = Calls()
    result = run(perfect, count=250, calls=calls)

    assert [len(body["items"]) for body in calls.bodies()] == [100, 100, 50]
    assert result.total == 250
    assert result.correct == 250
    assert result.failed_batches == 0
    assert result.batches == 3


def test_the_request_matches_the_canonical_wrapper_contract() -> None:
    calls = Calls()
    run(perfect, count=3, calls=calls)

    request = calls.requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://miner-a.chutes.ai/moderate"
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    assert request.headers["content-type"] == "application/json"

    body = calls.bodies()[0]
    assert body["policy"] == POLICY
    assert body["policy_version"] == POLICY_VERSION
    assert body["items"] == [
        {"id": "p000", "content": "content 0"},
        {"id": "p001", "content": "content 1"},
        {"id": "p002", "content": "content 2"},
    ]
    assert set(body) == {"policy", "policy_version", "items"}


def test_verdicts_are_matched_by_id_not_by_position() -> None:
    def shuffled(request: httpx.Request) -> httpx.Response:
        response = perfect(request)
        body = response.json()
        body["verdicts"] = list(reversed(body["verdicts"]))
        return httpx.Response(200, json=body)

    result = run(shuffled, count=50)
    assert result.correct == 50


def test_a_wrong_model_is_scored_wrong_not_failed() -> None:
    def inverted(request: httpx.Request) -> httpx.Response:
        body = response_body(perfect(request))
        body["verdicts"] = [
            {"id": verdict["id"], "violates": not verdict["violates"]}
            for verdict in body["verdicts"]
        ]
        return httpx.Response(200, json=body)

    result = run(inverted, count=100)
    assert result.correct == 0
    assert result.failed_batches == 0
    assert result.true_positives == 0
    assert result.false_positives + result.false_negatives == 100


def test_tokens_accumulate_across_batches() -> None:
    result = run(perfect, count=250)

    # 11 tokens per item across all 250 items, from the usage block.
    assert result.total_tokens == 250 * 11
    assert result.scored_token_items == 250
    assert result.mean_total_tokens == pytest.approx(11.0)


def test_no_items_means_no_requests() -> None:
    calls = Calls()
    result = run(perfect, count=0, calls=calls)

    assert len(calls) == 0
    assert result.total == 0
    assert result.batches == 0
    assert result.raw_accuracy == 0.0


# --- transport failure ------------------------------------------------------


def test_a_transport_error_is_retried_exactly_once_then_scored_incorrect() -> None:
    calls = Calls()

    def always_refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = run(always_refused, count=100, calls=calls)

    assert len(calls) == 2, "one attempt plus exactly one retry"
    assert result.failed_batches == 1
    assert result.total == 100
    assert result.correct == 0


def test_a_retry_that_succeeds_scores_the_batch_normally() -> None:
    calls = Calls()

    def flaky(request: httpx.Request) -> httpx.Response:
        if len(calls) == 1:
            raise httpx.ConnectError("first attempt dropped")
        return perfect(request)

    result = run(flaky, count=100, calls=calls)

    assert len(calls) == 2
    assert result.failed_batches == 0
    assert result.correct == 100


def test_retries_can_be_switched_off() -> None:
    calls = Calls()

    def always_refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    result = run(always_refused, count=100, calls=calls, transport_retries=0)

    assert len(calls) == 1
    assert result.failed_batches == 1


def test_a_timeout_is_not_retried() -> None:
    calls = Calls()

    def too_slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    result = run(too_slow, count=100, calls=calls)

    assert len(calls) == 1, "a timeout is the serving budget, not a blip"
    assert result.failed_batches == 1
    assert result.correct == 0
    assert result.total == 100


# --- contract failure -------------------------------------------------------


@pytest.mark.parametrize("status", [400, 401, 422, 429, 500, 503])
def test_a_non_2xx_fails_the_batch_without_retrying(status: int) -> None:
    calls = Calls()

    def refuses(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "no"})

    result = run(refuses, count=100, calls=calls)

    assert len(calls) == 1
    assert result.failed_batches == 1
    assert result.correct == 0


def broken(mutate: Callable[[dict[str, Any]], None]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        body = response_body(perfect(request))
        mutate(body)
        return httpx.Response(200, json=body)

    return handler


def drop_one(body: dict[str, Any]) -> None:
    body["verdicts"].pop()


def add_one(body: dict[str, Any]) -> None:
    body["verdicts"].append({"id": "extra", "violates": True})


def rename_one(body: dict[str, Any]) -> None:
    body["verdicts"][0]["id"] = "not-in-this-batch"


def duplicate_one(body: dict[str, Any]) -> None:
    body["verdicts"][1] = dict(body["verdicts"][0])


def integer_verdict(body: dict[str, Any]) -> None:
    body["verdicts"][0]["violates"] = 1


def string_verdict(body: dict[str, Any]) -> None:
    body["verdicts"][0]["violates"] = "YES"


def drop_usage(body: dict[str, Any]) -> None:
    del body["usage"]


def negative_usage(body: dict[str, Any]) -> None:
    body["usage"]["completion_tokens"] = -5


def string_usage(body: dict[str, Any]) -> None:
    body["usage"]["prompt_tokens"] = "many"


def drop_verdicts(body: dict[str, Any]) -> None:
    del body["verdicts"]


@pytest.mark.parametrize(
    "mutate",
    [
        drop_one,
        add_one,
        rename_one,
        duplicate_one,
        integer_verdict,
        string_verdict,
        drop_usage,
        negative_usage,
        string_usage,
        drop_verdicts,
    ],
    ids=lambda fn: fn.__name__,
)
def test_a_body_that_breaks_the_contract_fails_the_whole_batch(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    calls = Calls()
    result = run(broken(mutate), count=100, calls=calls)

    assert len(calls) == 1, "a contract violation is not a transport problem"
    assert result.failed_batches == 1
    assert result.correct == 0
    assert result.total == 100


def test_a_non_json_body_fails_the_batch() -> None:
    def html(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>gateway</html>")

    result = run(html, count=100)
    assert result.failed_batches == 1
    assert result.correct == 0


def test_a_json_array_body_fails_the_batch() -> None:
    def array(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    result = run(array, count=100)
    assert result.failed_batches == 1


# --- a failed batch is scored, never skipped --------------------------------


def test_a_failed_batch_scores_as_all_incorrect_rather_than_disappearing() -> None:
    result = run(failing_batch_handler(perfect, fail_batch=1), count=250)

    assert result.total == 250, "the failed batch's items still count"
    assert result.batches == 3
    assert result.failed_batches == 1
    assert result.failed_items == 100
    assert result.correct == 150, "only the two healthy batches earn credit"
    assert result.raw_accuracy == pytest.approx(0.6)


def test_a_failed_batch_lands_in_the_confusion_matrix_opposite_the_truth() -> None:
    result = run(failing_batch_handler(perfect, fail_batch=0), count=100)

    matrix = (
        result.true_positives,
        result.false_positives,
        result.true_negatives,
        result.false_negatives,
    )
    assert sum(matrix) == 100
    assert result.true_positives == 0
    assert result.true_negatives == 0
    # 50 positives went unanswered (misses), 50 negatives went unanswered.
    assert result.false_negatives == 50
    assert result.false_positives == 50


def test_a_failed_batch_is_recorded_per_item_with_no_verdict() -> None:
    result = run(failing_batch_handler(perfect, fail_batch=0), count=150)

    failed = [verdict for verdict in result.verdicts if verdict.batch_failed]
    assert len(failed) == 100
    assert all(verdict.predicted_violating is None for verdict in failed)
    assert all(not verdict.correct for verdict in failed)
    assert {verdict.batch_index for verdict in failed} == {0}

    answered = [verdict for verdict in result.verdicts if not verdict.batch_failed]
    assert len(answered) == 50
    assert all(verdict.predicted_violating is not None for verdict in answered)


def test_a_failed_batch_is_outside_the_token_mean() -> None:
    """Counting an unanswered batch as zero tokens would reward failing."""
    result = run(failing_batch_handler(perfect, fail_batch=1), count=250)

    assert result.scored_token_items == 150
    assert result.total_tokens == 150 * 11
    assert result.mean_total_tokens == pytest.approx(11.0)


def test_the_failure_reason_is_recorded_for_operators() -> None:
    result = run(failing_batch_handler(perfect, fail_batch=0), count=100)

    assert len(result.failure_reasons) == 1
    assert ModelResponseError.code in result.failure_reasons[0]


# --- caller mistakes fail before any request --------------------------------


def unreachable(request: httpx.Request) -> httpx.Response:
    raise AssertionError("no request should have been sent")


@pytest.fixture()
def client() -> Iterator[httpx.Client]:
    with httpx.Client(transport=httpx.MockTransport(unreachable)) as instance:
        yield instance


def evaluate(client: httpx.Client, **overrides: Any) -> MinerEvaluation:
    kwargs: dict[str, Any] = {
        "client": client,
        "target": TARGET,
        "items": items(10),
        "policy": POLICY,
        "policy_version": POLICY_VERSION,
        "api_key": API_KEY,
    }
    kwargs.update(overrides)
    return evaluate_miner(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"policy": ""}, "policy text"),
        ({"policy_version": ""}, "policy_version"),
        ({"api_key": ""}, "Chutes API key"),
        ({"transport_retries": -1}, "transport_retries"),
        ({"batch_size": 101}, "batch size"),
        ({"batch_size": 0}, "batch size"),
    ],
)
def test_caller_mistakes_raise_before_anything_is_sent(
    client: httpx.Client, overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(EvaluationError, match=match):
        evaluate(client, **overrides)


def test_a_plaintext_endpoint_is_refused() -> None:
    with pytest.raises(EvaluationError, match="https"):
        MinerTarget(
            hotkey="5FA0miner_a",
            chute_id="chute-abc",
            moderate_url="http://miner-a.chutes.ai/moderate",
        )


@pytest.mark.parametrize("field", ["hotkey", "chute_id", "moderate_url"])
def test_a_target_needs_every_field(field: str) -> None:
    kwargs = {
        "hotkey": "5FA0miner_a",
        "chute_id": "chute-abc",
        "moderate_url": "https://miner-a.chutes.ai/moderate",
    }
    kwargs[field] = ""
    with pytest.raises(EvaluationError, match=field):
        MinerTarget(**kwargs)


def test_a_smaller_batch_size_still_covers_every_item() -> None:
    calls = Calls()
    result = run(perfect, count=25, calls=calls, batch_size=10)

    assert [len(body["items"]) for body in calls.bodies()] == [10, 10, 5]
    assert result.total == 25
    assert result.correct == 25


# --- a miner must never be able to abort the cycle --------------------------


class TestAMinerCannotTakeDownTheCycle:
    """`evaluate_miner` promises never to raise for a model-side failure.

    Every miner is evaluated in the same 04:00 UTC cycle on every validator, so
    an exception that escapes here does not cost one miner its score. It costs
    the whole subnet its weight submission for that day, simultaneously,
    everywhere.
    """

    def test_a_body_that_lies_about_its_encoding_is_scored_not_raised(self) -> None:
        """The cheapest possible denial of service: one response header.

        `client.post` content-decodes the body inline, so declaring gzip over
        bytes that are not gzip raises `httpx.DecodingError`. That is a
        `RequestError` but *not* a `TransportError`, so a handler written
        against `TransportError` never sees it and it propagates out of
        `evaluate_miner` entirely.
        """
        calls = Calls()

        def bad_encoding(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-encoding": "gzip"}, content=b"not gzip at all"
            )

        with make_client(bad_encoding, calls) as client:
            result = evaluate(client, items=items(4))

        assert result.failed_batches == 1
        assert result.total == 4
        assert result.correct == 0
        assert any("could not be read" in reason for reason in result.failure_reasons)

    def test_a_redirect_loop_is_scored_not_raised(self) -> None:
        """`TooManyRedirects` is the same class of escape as `DecodingError`."""

        def redirect(request: httpx.Request) -> httpx.Response:
            return httpx.Response(307, headers={"location": str(request.url)})

        with httpx.Client(
            transport=httpx.MockTransport(redirect), follow_redirects=True, max_redirects=2
        ) as client:
            result = evaluate(client, items=items(4))

        assert result.failed_batches == 1
        assert result.correct == 0

    def test_a_model_that_fails_every_batch_reports_no_token_reading(self) -> None:
        """There is no reading at all. A zero would read as free inference."""
        calls = Calls()

        def broken(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="upstream is down")

        with make_client(broken, calls) as client:
            result = evaluate(client, items=items(6))

        assert result.has_token_reading is False
        assert result.mean_total_tokens is None
        assert result.mean_total_tokens_milli is None


class TestMinerControlledInputIsBounded:
    """The response is entirely miner-controlled and httpx has no size limit.

    The registry already caps the equally miner-controlled wrapper source at
    `MAX_SOURCE_BYTES`; this is the same hazard on the other surface.
    """

    def test_an_oversized_body_is_scored_not_swallowed(self) -> None:
        calls = Calls()
        oversized = b'{"padding":"' + b"x" * (MAX_RESPONSE_BYTES + 1024) + b'"}'

        def flood(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=oversized)

        with make_client(flood, calls) as client:
            result = evaluate(client, items=items(4))

        assert result.failed_batches == 1
        assert any("bytes" in reason for reason in result.failure_reasons)

    def test_a_conformant_body_is_nowhere_near_the_ceiling(self) -> None:
        """The cap must not be reachable by an honest model."""
        calls = Calls()
        with make_client(perfect, calls) as client:
            evaluate(client, items=items(100))
        assert len(calls.requests[0].content) < MAX_RESPONSE_BYTES // 100


class TestAnUnusableUrlIsACallerMistake:
    """Raised at construction, before any request, like every other caller error."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://",  # ValueError from urllib, mid-request
            "https:///moderate",  # no host
            "http://host/moderate",  # would leak the validator's API key
            "ftp://host/moderate",
        ],
    )
    def test_an_unusable_url_is_rejected_on_construction(self, url: str) -> None:
        with pytest.raises(EvaluationError):
            MinerTarget(hotkey="5F", chute_id="c", moderate_url=url)

    def test_a_real_url_still_constructs(self) -> None:
        assert MinerTarget(
            hotkey="5F", chute_id="c", moderate_url="https://guard.chutes.ai/moderate"
        ).moderate_url.endswith("/moderate")
