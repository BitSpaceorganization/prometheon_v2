"""A labelling harness that exercises the real client over a mock socket.

Nothing here stubs :class:`OpenAICompatibleClient`. Every batching test drives
the shipped client, the shipped prompt, and the shipped envelope through an
``httpx.MockTransport``, so a handler sees the bytes a provider would see and
the tests cannot agree with a mock that drifted from the code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest

from prometheon.config import LabellingConfig
from prometheon.labelling.client import OpenAICompatibleClient
from prometheon.labelling.prompt import ITEMS_BEGIN_PREFIX, ITEMS_END_PREFIX

#: Greedy, because an item may contain text that looks like the closing marker
#: and the *last* one is the real boundary. That forgery is exactly what the
#: production nonce makes impossible to attempt.
_ITEMS_RE = re.compile(
    rf"{ITEMS_BEGIN_PREFIX}(?P<fence>[0-9a-f]+)\n(?P<items>.*)\n{ITEMS_END_PREFIX}(?P=fence)",
    re.DOTALL,
)

#: What a handler may return: the assistant's text, or a whole HTTP response
#: when the test is about the transport rather than the answer.
Handler = Callable[[list[dict[str, str]]], "str | httpx.Response"]


def completion_envelope(
    text: str,
    *,
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """A chat-completions body shaped the way a provider shapes one."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_754_000_000,
        "model": "gpt-5.6-sol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sent_items(request: httpx.Request) -> list[dict[str, str]]:
    """The items a request actually carried, read back out of the prompt."""
    body = json.loads(request.content)
    user_prompt = body["messages"][1]["content"]
    match = _ITEMS_RE.search(user_prompt)
    if match is None:
        return []
    parsed: list[dict[str, str]] = json.loads(match.group("items"))
    return parsed


def sent_fence(request: httpx.Request) -> str:
    """The nonce that fenced a request's item block."""
    body = json.loads(request.content)
    match = _ITEMS_RE.search(body["messages"][1]["content"])
    assert match is not None, "the user prompt carried no fenced item block"
    return match.group("fence")


def verdicts(items: list[dict[str, str]], truth: dict[str, bool]) -> str:
    """A well-formed answer for every item, from a truth table."""
    return json.dumps(
        {"labels": [{"id": item["id"], "violates": truth[item["id"]]} for item in items]}
    )


class Harness:
    """One client, plus everything the transport saw."""

    def __init__(self, handler: Handler, config: LabellingConfig, **kwargs: Any) -> None:
        self.requests: list[httpx.Request] = []
        self.batches: list[list[str]] = []
        self.sleeps: list[float] = []

        def respond(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            items = sent_items(request)
            self.batches.append([item["id"] for item in items])
            outcome = handler(items)
            if isinstance(outcome, httpx.Response):
                return outcome
            return httpx.Response(200, json=completion_envelope(outcome))

        self.client = OpenAICompatibleClient(
            config,
            transport=httpx.MockTransport(respond),
            sleep=self.sleeps.append,
            **kwargs,
        )

    @property
    def calls(self) -> int:
        return len(self.requests)


@pytest.fixture
def build_labeller(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Harness]]:
    """Factory for a :class:`Harness` with a credential already in the env."""
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    built: list[Harness] = []

    def build(handler: Handler, *, config: LabellingConfig | None = None, **kwargs: Any) -> Harness:
        harness = Harness(handler, config or LabellingConfig(), **kwargs)
        built.append(harness)
        return harness

    yield build
    for harness in built:
        harness.client.close()
