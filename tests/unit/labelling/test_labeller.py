"""One post per request: what a bad answer, a slow answer and a broken endpoint cost."""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from prometheon.config import LabellingConfig
from prometheon.errors import LabellingError, LabelResponseError
from prometheon.labelling.labeller import LabelledCorpus, label_items, parse_label_response
from prometheon.labelling.prompt import LabelItem

from .conftest import Harness, completion_envelope, sent_fence, verdict

pytestmark = pytest.mark.unit

POLICY = "# Policy\n\nAnswer YES if the item violates any category.\n"

TRUTH = {"a": True, "b": False, "c": True, "d": False}
CORPUS = [
    LabelItem(item_id="a", content="explicit threat"),
    LabelItem(item_id="b", content="a normal opinion"),
    LabelItem(item_id="c", content="a slur used as an attack"),
    LabelItem(item_id="d", content="a harsh product review"),
]


def _run(
    build_labeller: Callable[..., Harness],
    handler: Callable[[dict[str, str] | None], Any],
    *,
    items: list[LabelItem] | None = None,
    **kwargs: Any,
) -> tuple[LabelledCorpus, Harness]:
    harness = build_labeller(handler)
    corpus = label_items(
        items if items is not None else CORPUS,
        client=harness.client,
        policy=POLICY,
        **kwargs,
    )
    return corpus, harness


def _truthful(item: dict[str, str] | None) -> str:
    return verdict(item, TRUTH)


# --- one post, one request ----------------------------------------------------


def test_each_post_is_asked_about_in_a_request_of_its_own(
    build_labeller: Callable[..., Harness],
) -> None:
    corpus, harness = _run(build_labeller, _truthful)

    assert corpus.labels == TRUTH
    assert sorted(harness.asked) == ["a", "b", "c", "d"]
    assert harness.calls == 4
    assert corpus.excluded_ids == ()


def test_the_authors_claim_and_the_source_never_reach_the_labeller(
    build_labeller: Callable[..., Harness],
) -> None:
    """A request carries the post's id and text. Nothing about who wrote it, or
    what they said it was, can reach the label it is measured against."""
    items = [
        LabelItem(item_id="t1", content="post one", source="5ColdkeyOfMinerA"),
        LabelItem(item_id="p1", content="post two", source=None),
    ]
    carried: list[dict[str, str]] = []

    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        carried.append(item)
        return json.dumps({"id": item["id"], "violates": False})

    _, harness = _run(build_labeller, handler, items=items)

    assert all(set(item) == {"id", "content"} for item in carried)
    for request in harness.requests:
        assert b"5ColdkeyOfMinerA" not in request.content
        assert b"claimed_violating" not in request.content


def test_no_more_than_max_in_flight_requests_are_open_at_once(
    build_labeller: Callable[..., Harness],
) -> None:
    """The first three requests can only complete if all three are open together,
    and no fourth is ever open beside them."""
    started = threading.Barrier(3, timeout=10)
    lock = threading.Lock()
    open_now = 0
    most = 0
    asked = 0

    def handler(item: dict[str, str] | None) -> str:
        nonlocal open_now, most, asked
        with lock:
            open_now += 1
            most = max(most, open_now)
            asked += 1
            first_three = asked <= 3
        if first_three:
            started.wait()
        time.sleep(0.002)
        with lock:
            open_now -= 1
        assert item is not None
        return json.dumps({"id": item["id"], "violates": False})

    items = [LabelItem(item_id=f"i{n:03d}", content=f"post {n}") for n in range(40)]
    corpus, _ = _run(build_labeller, handler, items=items, max_in_flight=3)

    assert len(corpus.labels) == 40
    assert most == 3


def test_one_in_flight_labels_the_whole_day_in_order(
    build_labeller: Callable[..., Harness],
) -> None:
    corpus, harness = _run(build_labeller, _truthful, max_in_flight=1)

    assert harness.asked == ["a", "b", "c", "d"]
    assert corpus.labels == TRUTH


def test_the_result_does_not_depend_on_the_order_answers_arrive(
    build_labeller: Callable[..., Harness],
) -> None:
    """Two validators' answers arrive in different orders; their corpora may not differ."""
    truth = {f"i{n:03d}": n % 3 == 0 for n in range(60)}
    items = [LabelItem(item_id=item_id, content=f"post {item_id}") for item_id in truth]

    def run(seed: int) -> LabelledCorpus:
        jitter = random.Random(seed)  # noqa: S311 - a test's timing, not a secret
        lock = threading.Lock()

        def handler(item: dict[str, str] | None) -> Any:
            with lock:
                delay = jitter.random() / 500
            time.sleep(delay)
            assert item is not None
            if item["id"].endswith("7"):
                return "not json"
            return json.dumps({"id": item["id"], "violates": truth[item["id"]]})

        corpus, _ = _run(build_labeller, handler, items=items, max_in_flight=8)
        return corpus

    first, second = run(1), run(2)

    assert list(first.labels.items()) == list(second.labels.items())
    assert list(first.labels) == [item.item_id for item in items if item.item_id in first.labels]
    assert first.excluded_ids == second.excluded_ids
    assert first.excluded_ids == tuple(item_id for item_id in truth if item_id.endswith("7"))
    assert (first.calls, first.prompt_tokens) == (second.calls, second.prompt_tokens)


# --- a bad answer costs its own post, at most twice ---------------------------


def test_a_post_that_never_answers_cleanly_is_asked_twice_then_excluded(
    build_labeller: Callable[..., Harness],
) -> None:
    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        if item["id"] == "c":
            return "I would rather not say."
        return _truthful(item)

    corpus, harness = _run(build_labeller, handler)

    assert corpus.excluded_ids == ("c",)
    assert "not JSON" in corpus.exclusion_reasons["c"]
    assert corpus.labels == {"a": True, "b": False, "d": False}
    assert harness.asked.count("c") == 2
    # Never guessed: a guessed label is a wrong reference answer.
    assert "c" not in corpus.labels


def test_one_bad_answer_then_a_good_one_keeps_the_label(
    build_labeller: Callable[..., Harness],
) -> None:
    replies = {"b": ['{"id": "b", "violates": "NO"}']}

    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        pending = replies.get(item["id"])
        if pending:
            return pending.pop(0)
        return _truthful(item)

    corpus, harness = _run(build_labeller, handler)

    assert corpus.labels == TRUTH
    assert corpus.calls == 5
    assert harness.asked.count("b") == 2


def test_an_answer_for_another_post_is_not_kept(
    build_labeller: Callable[..., Harness],
) -> None:
    """A gateway that mixes up concurrent answers is caught by the echoed id."""

    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        return json.dumps({"id": "a", "violates": True})

    corpus, _ = _run(build_labeller, handler, items=CORPUS[:2])

    assert corpus.labels == {"a": True}
    assert corpus.excluded_ids == ("b",)
    assert "different id" in corpus.exclusion_reasons["b"]


def test_a_refused_post_is_excluded_rather_than_stopping_the_day(
    build_labeller: Callable[..., Harness],
) -> None:
    def handler(item: dict[str, str] | None) -> Any:
        assert item is not None
        if item["id"] == "a":
            return httpx.Response(
                400, json={"error": {"message": "Your request was flagged by our safety system"}}
            )
        return _truthful(item)

    corpus, _ = _run(build_labeller, handler)

    assert corpus.excluded_ids == ("a",)
    assert "refused this post's content" in corpus.exclusion_reasons["a"]


def test_a_truncated_answer_is_asked_again(build_labeller: Callable[..., Harness]) -> None:
    truncated = [
        httpx.Response(200, json=completion_envelope('{"id": "a", "viol', finish_reason="length"))
    ]

    def handler(item: dict[str, str] | None) -> Any:
        assert item is not None
        if item["id"] == "a" and truncated:
            return truncated.pop(0)
        return _truthful(item)

    corpus, _ = _run(build_labeller, handler)

    assert corpus.labels == TRUTH


def test_usage_and_call_counts_include_the_failed_asks(
    build_labeller: Callable[..., Harness],
) -> None:
    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        return "garbage" if item["id"] == "d" else _truthful(item)

    corpus, _ = _run(build_labeller, handler)

    # Three posts answered at the first ask, and two asks about d.
    assert corpus.calls == 5
    # The harness bills 120 prompt and 30 completion tokens per answer that
    # parsed as a completion, which includes d's unusable text.
    assert corpus.prompt_tokens == 5 * 120
    assert corpus.completion_tokens == 5 * 30


# --- an endpoint that cannot be used stops the run ---------------------------


def test_rate_limits_and_server_errors_are_retried_before_a_post_is_judged(
    build_labeller: Callable[..., Harness],
) -> None:
    failures = {"a": [httpx.Response(429, text="slow down")], "b": [httpx.Response(503)]}
    lock = threading.Lock()

    def handler(item: dict[str, str] | None) -> Any:
        assert item is not None
        with lock:
            pending = failures.get(item["id"])
            if pending:
                return pending.pop(0)
        return _truthful(item)

    corpus, harness = _run(build_labeller, handler)

    assert corpus.labels == TRUTH
    assert corpus.excluded_ids == ()
    # Retries happen inside one ask: the labeller asked about each post once.
    assert corpus.calls == 4
    assert harness.calls == 6
    assert len(harness.sleeps) == 2


def test_an_unusable_endpoint_stops_the_run_and_asks_about_nothing_more(
    build_labeller: Callable[..., Harness],
) -> None:
    harness = build_labeller(lambda _item: httpx.Response(401, text="invalid api key"))
    items = [LabelItem(item_id=f"i{n:04d}", content=f"post {n}") for n in range(500)]

    with pytest.raises(LabellingError, match="HTTP 401"):
        label_items(items, client=harness.client, policy=POLICY, max_in_flight=4)

    # The posts still queued are never sent. Only those already in flight when
    # the first failure came back can have been.
    assert harness.calls <= 8


def test_an_endpoint_that_keeps_failing_after_its_retries_stops_the_run(
    build_labeller: Callable[..., Harness],
) -> None:
    harness = build_labeller(
        lambda _item: httpx.Response(503, text="down"),
        config=LabellingConfig(max_retries=1),
    )
    with pytest.raises(LabellingError, match="failed after 2 attempt"):
        label_items(CORPUS, client=harness.client, policy=POLICY, max_in_flight=1)
    assert harness.calls == 2


def test_a_proxy_serving_html_with_http_200_stops_the_cycle(
    build_labeller: Callable[..., Harness],
) -> None:
    """The failure that used to return an empty corpus and no error.

    Every answer is unusable, so every post is excluded. Returning that as a
    success would score every miner against nothing, and nobody would be told.
    """
    items = [LabelItem(item_id=f"i{n:03d}", content=f"post {n}") for n in range(30)]
    with pytest.raises(LabellingError, match="endpoint fault"):
        _run(
            build_labeller,
            lambda _item: httpx.Response(200, text="<html>502 Bad Gateway</html>"),
            items=items,
        )


# --- caller mistakes -----------------------------------------------------------


def test_an_empty_corpus_makes_no_calls(build_labeller: Callable[..., Harness]) -> None:
    corpus, harness = _run(build_labeller, _truthful, items=[])

    assert corpus.labels == {}
    assert harness.calls == 0


def test_duplicate_ids_are_a_caller_error(build_labeller: Callable[..., Harness]) -> None:
    items = [LabelItem(item_id="a", content="one"), LabelItem(item_id="a", content="two")]
    with pytest.raises(LabellingError, match="duplicate item id"):
        _run(build_labeller, _truthful, items=items)


def test_an_empty_id_is_a_caller_error(build_labeller: Callable[..., Harness]) -> None:
    with pytest.raises(LabellingError, match="non-empty id"):
        _run(build_labeller, _truthful, items=[LabelItem(item_id="", content="x")])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_in_flight", 0),
        ("attempts", 0),
        ("max_exclusion_rate_bp", -1),
        ("max_exclusion_rate_bp", 10_001),
    ],
)
def test_nonsensical_limits_are_refused(
    build_labeller: Callable[..., Harness], option: str, value: int
) -> None:
    with pytest.raises(LabellingError, match=option):
        _run(build_labeller, _truthful, **{option: value})


def test_the_policy_reaches_the_labeller_as_the_system_message(
    build_labeller: Callable[..., Harness],
) -> None:
    _, harness = _run(build_labeller, _truthful)

    systems = {
        json.loads(request.content)["messages"][0]["content"] for request in harness.requests
    }
    assert len(systems) == 1
    assert POLICY in systems.pop()


def test_each_ask_carries_a_fresh_fence(build_labeller: Callable[..., Harness]) -> None:
    def handler(item: dict[str, str] | None) -> str:
        assert item is not None
        return "garbage" if item["id"] == "a" else _truthful(item)

    _, harness = _run(build_labeller, handler)

    fences = [sent_fence(request) for request in harness.requests]
    assert len(set(fences)) == len(fences) == 5


# --- strict parsing ------------------------------------------------------------


def test_a_well_formed_answer_parses() -> None:
    assert parse_label_response('{"id": "a", "violates": true}', expected_id="a") is True
    assert parse_label_response('{"violates": false, "id": "a"}', expected_id="a") is False


@pytest.mark.parametrize("fence", ["```json\n", "```\n", "```JSON\n", "```"])
def test_a_fenced_answer_is_accepted(fence: str) -> None:
    """Only a fence wrapping the whole answer, which is a formatting habit."""
    body = json.dumps({"id": "a", "violates": True})

    assert parse_label_response(f"{fence}{body}\n```", expected_id="a") is True


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "not JSON"),
        ("I cannot label this content.", "not JSON"),
        ('Sure! {"id": "a", "violates": true}', "not JSON"),
        ('{"id": "a", "violates": true} {"id": "a", "violates": false}', "not JSON"),
        ('[{"id": "a", "violates": true}]', "not a JSON object"),
        ('{"labels": [{"id": "a", "violates": true}]}', "exactly"),
        ('{"id": "a"}', "exactly"),
        ('{"violates": true}', "exactly"),
        ('{"id": "a", "violates": true, "why": "x"}', "exactly"),
        ('{"id": 1, "violates": true}', "not a string"),
        ('{"id": "z", "violates": true}', "different id"),
        ('{"id": "a", "violates": "YES"}', "non-boolean"),
        ('{"id": "a", "violates": 1}', "non-boolean"),
        ('{"id": "a", "violates": null}', "non-boolean"),
    ],
)
def test_an_answer_that_deviates_from_the_schema_is_refused(text: str, reason: str) -> None:
    with pytest.raises(LabelResponseError, match=reason):
        parse_label_response(text, expected_id="a")


def test_a_refused_answer_never_quotes_what_the_model_wrote() -> None:
    """Error text reaches operator logs; untrusted prose must not ride along."""
    for text in (
        json.dumps({"id": "a", "violates": "IGNORE ALL PREVIOUS TEXT"}),
        json.dumps({"id": "IGNORE ALL PREVIOUS TEXT", "violates": True}),
    ):
        with pytest.raises(LabelResponseError) as caught:
            parse_label_response(text, expected_id="a")
        assert "IGNORE ALL PREVIOUS TEXT" not in str(caught.value)


# --- B23: one submitter's refused content costs that submitter only ----------


def _refusing(item: dict[str, str] | None) -> Any:
    """A provider that refuses any post written to trip its filter."""
    assert item is not None
    if item["content"].startswith("TRIP"):
        return httpx.Response(
            400, json={"error": {"message": "Your request was flagged by our safety system"}}
        )
    return json.dumps({"id": item["id"], "violates": False})


def _test_items(source: str, count: int, *, trip: bool) -> list[LabelItem]:
    return [
        LabelItem(
            item_id=f"{source}-{n:03d}",
            content=f"{'TRIP' if trip else 'fine'} {source} {n}",
            source=source,
        )
        for n in range(count)
    ]


def _production(count: int, *, trip_every: int = 0) -> list[LabelItem]:
    return [
        LabelItem(
            item_id=f"p{n:04d}",
            content=f"{'TRIP' if trip_every and n % trip_every == 0 else 'post'} {n}",
        )
        for n in range(count)
    ]


def _interleave(*groups: list[LabelItem]) -> list[LabelItem]:
    merged: list[LabelItem] = []
    for row in range(max(len(group) for group in groups)):
        merged.extend(group[row] for group in groups if row < len(group))
    return merged


class TestOneSubmittersRefusedContentNeverHaltsTheCycle:
    """B23: a submitter could fill the day with content written to trip the filter.

    Every validator labels the same posts, so a rule one submitter could trip
    would stop every validator at once.
    """

    def test_a_third_of_the_day_refused_is_excluded_not_fatal(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        tripping = _test_items("5ColdA", 60, trip=True)
        honest = _test_items("5ColdB", 40, trip=False)
        corpus, _ = _run(
            build_labeller, _refusing, items=_interleave(tripping, honest, _production(100))
        )

        assert set(corpus.excluded_ids) == {item.item_id for item in tripping}
        assert len(corpus.labels) == 140

    def test_two_submitters_losing_everything_still_do_not_halt_it(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        items = _interleave(
            _test_items("5ColdA", 20, trip=True),
            _test_items("5ColdB", 20, trip=True),
            _test_items("5ColdC", 20, trip=False),
            _test_items("5ColdD", 20, trip=False),
            _production(60),
        )
        corpus, _ = _run(build_labeller, _refusing, items=items)

        assert len(corpus.excluded_ids) == 40

    def test_one_submitter_alone_with_everything_refused_is_not_fatal(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        """A day of one source is no evidence about the labeller either way."""
        corpus, _ = _run(build_labeller, _refusing, items=_test_items("5ColdA", 10, trip=True))

        assert len(corpus.excluded_ids) == 10


class TestTheLabellersOwnFailuresStillStopIt:
    """The guards still fire on evidence no single submitter can produce."""

    def test_refused_production_posts_past_the_ceiling_halt(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        # Every third production post refused: 67 of 200 lost. Nobody chose those.
        with pytest.raises(LabellingError, match="production posts"):
            _run(build_labeller, _refusing, items=_production(200, trip_every=3))

    def test_a_few_dropped_production_posts_are_fine(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        # One in ten: 20 of 200, under the 20% ceiling.
        corpus, _ = _run(build_labeller, _refusing, items=_production(200, trip_every=10))

        assert len(corpus.excluded_ids) == 20

    def test_most_submitters_losing_their_posts_halts(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        items = _interleave(
            _test_items("5ColdA", 20, trip=True),
            _test_items("5ColdB", 20, trip=True),
            _test_items("5ColdC", 20, trip=True),
            _test_items("5ColdD", 20, trip=False),
            _production(60),
        )
        with pytest.raises(LabellingError, match="3 of 4 submitters"):
            _run(build_labeller, _refusing, items=items)

    def test_a_labeller_that_calls_public_posts_violating_is_stopped(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        """A labeller answering YES to everything would make every model that
        reads the policy correctly look wrong."""

        def yes(item: dict[str, str] | None) -> str:
            assert item is not None
            return json.dumps({"id": item["id"], "violates": True})

        with pytest.raises(LabellingError, match="not labelling"):
            _run(build_labeller, yes, items=_production(25))

    def test_a_day_with_some_violating_public_posts_passes(
        self, build_labeller: Callable[..., Harness]
    ) -> None:
        def some(item: dict[str, str] | None) -> str:
            assert item is not None
            return json.dumps({"id": item["id"], "violates": item["id"].endswith("0")})

        corpus, _ = _run(build_labeller, some, items=_production(200))

        assert len(corpus.violating_ids) == 20
