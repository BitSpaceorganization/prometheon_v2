"""Ground truth, one post per request.

Every post is labelled in a request of its own, carrying the policy, the post's
id and its text, and nothing else. The corpus is *sourced* to contain policy
violations, so some of it is written to attack whatever reads it. Alone in a
request, whatever a post carries can decide only its own verdict.

What happens to a post depends on how its request failed:

- **A bad answer**, meaning not JSON, the wrong id, a verdict that is not a
  boolean, a truncated reply, a refusal, or the provider declining the post's
  content: the post is asked once more, then excluded. Nothing is guessed. A
  guessed label is a wrong reference answer that silently penalises the models
  that got it right. An excluded test post still costs its submitter: it
  stays in their ``S`` and counts in neither ``V``.
- **Rate limits and server errors** are retried with backoff by the client,
  :mod:`prometheon.labelling.client`, before they reach here.
- **An unusable endpoint**, meaning a rejected key, an unknown model, a
  malformed request, or retries exhausted: the run stops at once, because
  nothing about the next post would fix it.

**A day's exclusions are judged as a whole.** Dropping a few posts is harmless
only at a low rate, and no single submitter may decide whether the cycle runs
(B23). So the run stops only on evidence no one submitter can produce:

- every post excluded, across more than one source;
- more than ``max_exclusion_rate_bp`` of the production posts excluded, which
  nobody submitted;
- most of the test posts lost for at least three submitters, and for at least
  half of them;
- more than 90% of at least 20 labelled production posts coming back
  violating. Public posts sampled by the partner are not like that, so the
  labeller has stopped labelling.

**Up to ``max_in_flight`` requests run at once.** Answers are collected by post,
not by arrival, so the labels, the exclusions and every count come out the
same whatever order the answers arrive in.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Final

from prometheon.errors import LabellingError, LabelResponseError
from prometheon.labelling.client import ChatCompletionClient
from prometheon.labelling.prompt import (
    ID_KEY,
    VERDICT_KEY,
    LabelItem,
    build_system_prompt,
    build_user_prompt,
    new_fence,
)

#: Requests open at once. Enough to keep a hosted endpoint busy without
#: running into its per-key rate limit on an ordinary day.
DEFAULT_MAX_IN_FLIGHT: Final[int] = 8

#: Asks per post before it is excluded. One repeat covers a sampling hiccup;
#: more would let one genuinely unlabellable post cost a growing share of the
#: day's calls.
DEFAULT_ATTEMPTS: Final[int] = 2

#: Basis-point denominator, matching :mod:`prometheon.scoring.allocation`.
BASIS_POINTS: Final[int] = 10_000

#: The most of the production posts that may be dropped and still be called
#: ground truth. 20% is generous: a floor under "this is not a benchmark", not
#: a quality target.
DEFAULT_MAX_EXCLUSION_RATE_BP: Final[int] = 2_000

#: The fewest production posts the exclusion *rate* is meaningful over. Under
#: this, one dropped post is already a large percentage and says nothing about
#: the labeller.
_MIN_PRODUCTION_FOR_RATE_CHECK: Final[int] = 50

#: How many submitters must each have lost most of their test posts before
#: those losses say something about the labeller rather than the content. One
#: submitter's content costs that submitter, never the cycle (B23).
_MIN_SOURCES_FOR_SYSTEMIC_LOSS: Final[int] = 3

#: At least this many labelled production posts, and more than this share of
#: them violating, is a labeller that stopped labelling.
_MIN_PRODUCTION_FOR_DAY_CHECK: Final[int] = 20
_MAX_PRODUCTION_VIOLATING_BP: Final[int] = 9_000

#: Models wrap JSON in a fence often enough that refusing it would cost a
#: second ask for many posts. Only a fence around the *whole* response is
#: stripped. Prose, a second object, or a preamble is still refused.
_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"\A\s*```[a-zA-Z0-9_-]*\s*(?P<body>.*?)\s*```\s*\Z", re.DOTALL
)

#: How many key names an error message lists before it stops. Error text
#: reaches operator logs, so it names keys and ids and never content.
_MAX_REPORTED_KEYS: Final[int] = 5


@dataclass(frozen=True)
class LabelledCorpus:
    """Ground truth for a corpus, plus what it could not label.

    ``excluded_ids`` is part of the result rather than a log line: the day's
    evaluation must be reproducible from what the validator publishes, and that
    is impossible if posts vanished for reasons only one process saw.
    """

    labels: Mapping[str, bool]
    excluded_ids: tuple[str, ...]
    exclusion_reasons: Mapping[str, str]
    #: Asks, not HTTP requests: the client retries a rate limit or a server
    #: error inside one ask.
    calls: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def violating_ids(self) -> tuple[str, ...]:
        return tuple(item_id for item_id, violates in self.labels.items() if violates)


@dataclass(frozen=True)
class _Outcome:
    """One post's result: a verdict, or why there is none."""

    verdict: bool | None
    reason: str
    calls: int
    prompt_tokens: int
    completion_tokens: int


class _Stopped(Exception):  # noqa: N818 - control flow inside this module, never raised out
    """Another post found the endpoint unusable; this one was never asked."""


def label_items(
    items: Sequence[LabelItem],
    *,
    client: ChatCompletionClient,
    policy: str,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    attempts: int = DEFAULT_ATTEMPTS,
    fence_factory: Callable[[], str] = new_fence,
    max_exclusion_rate_bp: int = DEFAULT_MAX_EXCLUSION_RATE_BP,
) -> LabelledCorpus:
    """Label every post, each in a request of its own.

    Returns the labels obtained and the ids excluded, both in the order the
    posts were given. Raises :class:`LabellingError` for a caller mistake, an
    unusable endpoint, or a day whose exclusions are the labeller's doing
    rather than any one submitter's; see the module docstring.
    """
    if max_in_flight < 1:
        raise LabellingError(f"max_in_flight must be at least 1, got {max_in_flight}")
    if attempts < 1:
        raise LabellingError(f"attempts must be at least 1, got {attempts}")
    if not 0 <= max_exclusion_rate_bp <= BASIS_POINTS:
        raise LabellingError(
            f"max_exclusion_rate_bp must be within [0, {BASIS_POINTS}], got {max_exclusion_rate_bp}"
        )
    _require_labellable(items)
    if not items:
        return LabelledCorpus(
            labels={},
            excluded_ids=(),
            exclusion_reasons={},
            calls=0,
            prompt_tokens=0,
            completion_tokens=0,
        )

    # Rendered once: the policy runs to thousands of tokens and does not
    # change inside a cycle.
    system_prompt = build_system_prompt(policy)
    outcomes: dict[str, _Outcome] = {}
    # Set by the first post that finds the endpoint unusable. Every worker
    # checks it before each ask, so a broken endpoint costs the requests
    # already in flight and nothing more, however fast the others are failing.
    stop = threading.Event()
    pool = ThreadPoolExecutor(
        max_workers=min(max_in_flight, len(items)), thread_name_prefix="labeller"
    )
    try:
        futures: dict[Future[_Outcome], str] = {
            pool.submit(
                _label_one,
                item,
                client=client,
                system_prompt=system_prompt,
                attempts=attempts,
                fence_factory=fence_factory,
                stop=stop,
            ): item.item_id
            for item in items
        }
        for future in as_completed(futures):
            try:
                outcomes[futures[future]] = future.result()
            except _Stopped:
                # The post that stopped the run raises its own error, which
                # is the one worth reporting; it arrives in its own turn.
                continue
    finally:
        # On the way out of an unusable endpoint, the posts still queued are
        # never asked: without `cancel_futures` the pool would send every one
        # of them before letting the error through.
        pool.shutdown(wait=True, cancel_futures=True)

    ordered = [item.item_id for item in items]
    labels: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for item_id in ordered:
        outcome = outcomes[item_id]
        if outcome.verdict is None:
            reasons[item_id] = outcome.reason
        else:
            labels[item_id] = outcome.verdict

    _assert_exclusion_within_budget(items, reasons, max_rate_bp=max_exclusion_rate_bp)
    _assert_production_is_plausible(items, labels)
    return LabelledCorpus(
        labels=labels,
        excluded_ids=tuple(reasons),
        exclusion_reasons=reasons,
        calls=sum(outcome.calls for outcome in outcomes.values()),
        prompt_tokens=sum(outcome.prompt_tokens for outcome in outcomes.values()),
        completion_tokens=sum(outcome.completion_tokens for outcome in outcomes.values()),
    )


def _label_one(
    item: LabelItem,
    *,
    client: ChatCompletionClient,
    system_prompt: str,
    attempts: int,
    fence_factory: Callable[[], str],
    stop: threading.Event,
) -> _Outcome:
    """Ask about one post up to ``attempts`` times; a bad answer is never kept.

    A :class:`LabellingError` that is not a bad answer, an endpoint that cannot
    be used at all, sets ``stop`` and is raised to the caller.
    """
    calls = prompt_tokens = completion_tokens = 0
    failure = "no attempt was made"
    for _ in range(attempts):
        if stop.is_set():
            raise _Stopped
        # A fresh fence per ask, so a repeat cannot be steered by anything the
        # first ask revealed.
        prompt = build_user_prompt(item, fence=fence_factory())
        # Counted before the call: a truncated reply or a refusal is raised
        # after the provider billed it, and a counter that skipped those would
        # read lowest exactly when a day cost the most.
        calls += 1
        try:
            response = client.complete(system_prompt=system_prompt, user_prompt=prompt)
        except LabelResponseError as exc:
            failure = str(exc)
            continue
        except BaseException:
            stop.set()
            raise
        prompt_tokens += response.prompt_tokens
        completion_tokens += response.completion_tokens
        try:
            verdict = parse_label_response(response.text, expected_id=item.item_id)
        except LabelResponseError as exc:
            failure = str(exc)
            continue
        return _Outcome(verdict, "", calls, prompt_tokens, completion_tokens)
    return _Outcome(None, failure, calls, prompt_tokens, completion_tokens)


def parse_label_response(text: str, *, expected_id: str) -> bool:
    """Parse and fully validate one answer: ``{"id": <the post's id>, "violates": <bool>}``.

    Every deviation from the requested shape is refused, including harmless
    ones such as an extra key. A model that has started improvising the format
    has stopped following the instructions that also told it to ignore the
    content, and its verdict has lost the only evidence it had.
    """
    match = _FENCE_RE.match(text)
    body = match.group("body") if match else text.strip()
    try:
        payload: object = json.loads(body)
    except ValueError as exc:
        raise LabelResponseError(f"the answer is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelResponseError("the answer is not a JSON object")
    if set(payload) != {ID_KEY, VERDICT_KEY}:
        raise LabelResponseError(
            f"the answer must hold exactly {ID_KEY!r} and {VERDICT_KEY!r}, got "
            f"{sorted(map(str, payload))[:_MAX_REPORTED_KEYS]}"
        )
    answered: object = payload[ID_KEY]
    if not isinstance(answered, str):
        raise LabelResponseError(f"the answer's {ID_KEY!r} is not a string")
    if answered != expected_id:
        # The id it named is not quoted: it is model output, and model output
        # can carry the post's text into an operator's log.
        raise LabelResponseError(
            f"the answer names a different id from the post it was asked about ({expected_id!r})"
        )
    violates: object = payload[VERDICT_KEY]
    # bool first: `isinstance(True, int)` is true, and a verdict of 1 is a model
    # that answered in a different type system than the one asked for.
    if not isinstance(violates, bool):
        raise LabelResponseError(f"the answer for {expected_id!r} has a non-boolean verdict")
    return violates


def _require_labellable(items: Sequence[LabelItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if not item.item_id:
            raise LabellingError("every item needs a non-empty id to label against")
        if item.item_id in seen:
            # Two posts sharing an id would let one verdict overwrite another.
            raise LabellingError(f"duplicate item id in the corpus: {item.item_id!r}")
        seen.add(item.item_id)


def _sources(items: Sequence[LabelItem]) -> int:
    """How many independent sources ``items`` came from.

    A test post's ``source`` is its miner's coldkey, so everything one party
    submits is one source, however many posts or hotkeys it spans. Production
    content is a random sample with no submitter, so each post stands alone.
    """
    submitters = {item.source for item in items if item.source is not None}
    return len(submitters) + sum(1 for item in items if item.source is None)


def _last_reason(items: Sequence[LabelItem], reasons: Mapping[str, str]) -> str:
    """The reason the last excluded post, in corpus order, was dropped."""
    for item in reversed(items):
        if item.item_id in reasons:
            return reasons[item.item_id] or "none recorded"
    return "none recorded"


def _assert_exclusion_within_budget(
    items: Sequence[LabelItem], reasons: Mapping[str, str], *, max_rate_bp: int
) -> None:
    """Refuse a corpus too damaged to be a benchmark, but never over one submitter.

    **What one submitter's content does is not evidence** (B23). A provider
    refuses some content outright, and the corpus is sourced to contain policy
    violations, so a submitter who fills the day with content written to trip
    the filter would otherwise halt every validator at once. Their refused
    posts cost them alone.
    """
    if not items or not reasons:
        return
    last = _last_reason(items, reasons)

    # Nothing survived. There is no reading of "labelled nothing" that is a
    # usable ground truth, and returning it would score every miner against
    # nothing.
    if len(reasons) >= len(items) and _sources(items) >= 2:
        raise LabellingError(
            f"labelling excluded every one of {len(items)} posts, so the day has no ground "
            "truth. This is an endpoint fault rather than a content fault: a proxy answering "
            "HTTP 200 with an error page, a model ignoring the requested response format, or "
            f"a provider-side filter refusing everything. Last reason: {last}"
        )

    production = [item for item in items if item.source is None]
    if len(production) >= _MIN_PRODUCTION_FOR_RATE_CHECK:
        lost = sum(1 for item in production if item.item_id in reasons)
        rate_bp = (lost * BASIS_POINTS) // len(production)
        if rate_bp > max_rate_bp:
            raise LabellingError(
                f"labelling excluded {lost} of {len(production)} production posts "
                f"({rate_bp / 100:.1f}%), over the {max_rate_bp / 100:.1f}% ceiling. No "
                "submitter chose that content, so this is the labeller, and a corpus this "
                f"incomplete is not a benchmark. Last reason: {last}"
            )

    by_source: dict[str, list[str]] = {}
    for item in items:
        if item.source is not None:
            by_source.setdefault(item.source, []).append(item.item_id)
    mostly_lost = [
        source
        for source, ids in by_source.items()
        if sum(1 for item_id in ids if item_id in reasons) * 2 > len(ids)
    ]
    if len(mostly_lost) >= max(_MIN_SOURCES_FOR_SYSTEMIC_LOSS, (len(by_source) + 1) // 2):
        raise LabellingError(
            f"labelling lost most of the test posts of {len(mostly_lost)} of {len(by_source)} "
            "submitters. One submitter's content costs only that submitter, but losses this "
            f"widespread are the labeller's. Last reason: {last}"
        )


def _assert_production_is_plausible(items: Sequence[LabelItem], labels: Mapping[str, bool]) -> None:
    """Stop when the day's production posts come back implausibly violating.

    Production content is public posts sampled by the partner, chosen by no
    participant, and mostly compliant. A labeller that says nearly all of them
    violate has stopped labelling, and its answers must not become anyone's
    reference.
    """
    production = [item.item_id for item in items if item.source is None and item.item_id in labels]
    if len(production) < _MIN_PRODUCTION_FOR_DAY_CHECK:
        return
    violating = sum(1 for item_id in production if labels[item_id])
    if violating * BASIS_POINTS > _MAX_PRODUCTION_VIOLATING_BP * len(production):
        raise LabellingError(
            f"{violating} of {len(production)} production posts came back violating. Public "
            "posts sampled by the partner are not like that: the labeller is not labelling. "
            "Refusing to publish it as ground truth"
        )


__all__ = [
    "BASIS_POINTS",
    "DEFAULT_ATTEMPTS",
    "DEFAULT_MAX_EXCLUSION_RATE_BP",
    "DEFAULT_MAX_IN_FLIGHT",
    "LabelledCorpus",
    "label_items",
    "parse_label_response",
]
