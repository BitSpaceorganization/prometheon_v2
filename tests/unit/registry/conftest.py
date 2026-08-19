"""Shared fixtures for the registry tests.

Every request goes through ``httpx.MockTransport``, so what a test asserts on
is the request that would have gone on the wire, path and headers included,
rather than a stub agreeing with the code that called it. Nothing here opens a
socket or touches a file.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

Handler = Callable[[httpx.Request], httpx.Response]

REPO = "prometheon-labs/moderation-guard"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40

#: A repository that satisfies the manifest, including sharded weights.
#: Weights, config and tokenizer. No executable code: the engine belongs to the
#: validator now, so a repository is only ever read for model files.
CONFORMING_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Calls:
    """Records every request a client actually sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]

    def __len__(self) -> int:
        return len(self.requests)


MINER_HOTKEY = "5MinerOne"

#: What a conforming checkpoint declares itself to be. `qwen2` is deliberate:
#: it is a real architecture the pinned transformers knows, so a fixture using
#: it exercises the accepting path rather than the "unknown, do not guess" one.
MODEL_TYPE = "qwen2"

#: Comfortably under the ceiling, so size is not what a test trips over unless
#: it means to.
SHARD_BYTES = 4 * 1024**3


def revision_payload(
    *, files: tuple[str, ...] = CONFORMING_FILES, sha: str = REVISION
) -> dict[str, object]:
    """The shape the Hugging Face revision endpoint answers with."""
    return {
        "id": REPO,
        "sha": sha,
        "siblings": [
            {
                "rfilename": name,
                # Only weight shards carry a size worth counting, and the live
                # API only reports sizes when asked with `?blobs=true`.
                "size": SHARD_BYTES if name.endswith(".safetensors") else 512,
            }
            for name in files
        ],
    }


def hf_handler(
    *,
    files: tuple[str, ...] = CONFORMING_FILES,
    sha: str = REVISION,
    calls: Calls | None = None,
    model_type: str = MODEL_TYPE,
    shard_bytes: int = SHARD_BYTES,
) -> Handler:
    """A Hugging Face stand-in serving one repository at one revision."""
    config_json = json.dumps({"model_type": model_type}).encode() if model_type else b"{}"

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.requests.append(request)
        path = request.url.path
        if path.startswith("/api/models/") and "/revision/" in path:
            payload = revision_payload(files=files, sha=sha)
            for sibling in payload["siblings"]:  # type: ignore[index]
                if sibling["rfilename"].endswith(".safetensors"):
                    sibling["size"] = shard_bytes
            return httpx.Response(200, json=payload)
        if "/resolve/" in path:
            return httpx.Response(200, content=config_json)
        return httpx.Response(404, json={"error": "not found"})

    return handle


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    """Retries must never actually wait in a unit test."""

    def sleep(_seconds: float) -> None:
        return None

    return sleep
