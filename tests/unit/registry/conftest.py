"""Shared fixtures for the registry tests.

Every request goes through ``httpx.MockTransport``, so what a test asserts on
is the request that would have gone on the wire, path and headers included,
rather than a stub agreeing with the code that called it. Nothing here opens a
socket or touches a file.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from prometheon.canonical.hashes import (
    IMAGE_NAME,
    IMAGE_TAG,
    IMAGE_USERNAME,
    image_id_for,
)
from prometheon.canonical.integrity import render_wrapper

Handler = Callable[[httpx.Request], httpx.Response]

REPO = "prometheon-labs/moderation-guard"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40
CHUTE_ID = "b4e8a2f1-6c3d-4e5a-9b7f-1d2c3e4f5a6b"
SLUG = "moderation-guard"

#: A repository that satisfies the manifest, including sharded weights.
#: Weights, config and tokenizer. No executable code: the engine is in the
#: deploy script now, so a repository is only ever read for model files.
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


#: The account whose chute the fixtures describe, and the hotkey that names it.
CHUTES_USER = "miner-one"
MINER_HOTKEY = "5MinerOne"


def wrapper_source(
    *,
    hf_repo: str = REPO,
    hf_revision: str = REVISION,
    chutes_user: str = CHUTES_USER,
    hotkey: str = MINER_HOTKEY,
    image_id: str | None = None,
) -> str:
    return render_wrapper(
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        chutes_user=chutes_user,
        hotkey=hotkey,
        image_id=image_id or image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG),
    )


def revision_payload(
    *, files: tuple[str, ...] = CONFORMING_FILES, sha: str = REVISION
) -> dict[str, object]:
    """The shape the Hugging Face revision endpoint answers with."""
    return {
        "id": REPO,
        "sha": sha,
        "siblings": [{"rfilename": name} for name in files],
    }


def hf_handler(
    *,
    files: tuple[str, ...] = CONFORMING_FILES,
    sha: str = REVISION,
    engine: bytes | None = None,
    calls: Calls | None = None,
) -> Handler:
    """A Hugging Face stand-in serving one repository at one revision."""
    body = engine or b""

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.requests.append(request)
        path = request.url.path
        if path.startswith("/api/models/") and "/revision/" in path:
            return httpx.Response(200, json=revision_payload(files=files, sha=sha))
        if "/resolve/" in path:
            return httpx.Response(200, content=body)
        return httpx.Response(404, json={"error": "not found"})

    return handle


def chutes_handler(
    *,
    slug: str = SLUG,
    hot: bool = True,
    source: str | None = None,
    inline_code: bool = False,
    calls: Calls | None = None,
    extra_info: dict[str, object] | None = None,
    image_id: str | None = None,
    image_username: str = IMAGE_USERNAME,
) -> Handler:
    """A Chutes stand-in serving one chute and its deployed source."""
    deployed = wrapper_source() if source is None else source

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.requests.append(request)
        path = request.url.path
        if path == f"/chutes/code/{CHUTE_ID}":
            return httpx.Response(200, content=deployed.encode("utf-8"))
        if path == f"/chutes/{CHUTE_ID}":
            info: dict[str, object] = {
                "chute_id": CHUTE_ID,
                "name": "moderation-guard",
                "slug": slug,
                "hot": hot,
                # Shaped like the live API: the image is a nested object whose
                # owner is under `user.username`.
                "image": {
                    "image_id": (
                        image_id
                        if image_id is not None
                        else image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG)
                    ),
                    "user": {"username": image_username},
                },
            }
            if inline_code:
                info["code"] = deployed
            if extra_info:
                info.update(extra_info)
            return httpx.Response(200, json=info)
        return httpx.Response(404, json={"error": "not found"})

    return handle


@pytest.fixture
def no_sleep() -> Callable[[float], None]:
    """Retries must never actually wait in a unit test."""

    def sleep(_seconds: float) -> None:
        return None

    return sleep
