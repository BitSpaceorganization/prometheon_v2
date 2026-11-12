"""A labelling harness that exercises the real client over a mock socket.

Nothing here stubs :class:`OpenAICompatibleClient`. Every labelling test drives
the shipped client, the shipped prompt, and the shipped envelope through an
``httpx.MockTransport``, so a handler sees the bytes a provider would see and
the tests cannot agree with a mock that drifted from the code. The labeller
calls the client from several threads, so the harness records under a lock.
"""

from __future__ import annotations

import json
import re
import threading
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
_ITEM_RE = re.compile(
    rf"{ITEMS_BEGIN_PREFIX}(?P<fence>[0-9a-f]+)\n(?P<item>.*)\n{ITEMS_END_PREFIX}(?P=fence)",
    re.DOTALL,
)

#: What a handler is given: the item a request carried, or ``None`` when the
#: test sent a prompt with no item block. What it may return: the assistant's
#: text, or a whole HTTP response when the test is about the transport.
Handler = Callable[["dict[str, str] | None"], "str | httpx.Response"]


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


def sent_item(request: httpx.Request) -> dict[str, str] | None:
    """The item a request actually carried, read back out of the prompt."""
    body = json.loads(request.content)
    user_prompt = body["messages"][1]["content"]
    match = _ITEM_RE.search(user_prompt)
    if match is None:
        return None
    parsed: dict[str, str] = json.loads(match.group("item"))
    return parsed


def sent_fence(request: httpx.Request) -> str:
    """The nonce that fenced a request's item block."""
    body = json.loads(request.content)
    match = _ITEM_RE.search(body["messages"][1]["content"])
    assert match is not None, "the user prompt carried no fenced item block"
    return match.group("fence")


def verdict(item: dict[str, str] | None, truth: dict[str, bool]) -> str:
    """A well-formed answer for the item, from a truth table."""
    assert item is not None, "the request carried no item"
    return json.dumps({"id": item["id"], "violates": truth[item["id"]]})


class Harness:
    """One client, plus everything the transport saw."""

    def __init__(self, handler: Handler, config: LabellingConfig, **kwargs: Any) -> None:
        self.requests: list[httpx.Request] = []
        #: The id each request asked about, in the order requests arrived.
        self.asked: list[str] = []
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

        def respond(request: httpx.Request) -> httpx.Response:
            item = sent_item(request)
            with self._lock:
                self.requests.append(request)
                if item is not None:
                    self.asked.append(item["id"])
            outcome = handler(item)
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
