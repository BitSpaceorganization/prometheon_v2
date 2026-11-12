"""Reading P from safetensors headers: the format, the limits, and the range reads."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import httpx
import pytest

from prometheon.errors import ManifestViolationError, RegistryUnavailableError
from prometheon.registry.huggingface import HuggingFaceClient, RepoFile, RepoSnapshot
from prometheon.registry.safetensors import MAX_HEADER_BYTES, count_parameters, header_length

pytestmark = pytest.mark.unit

WHERE = "model.safetensors in org/name@" + "a" * 40


def _header(**shapes: list[int]) -> bytes:
    body: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, shape in shapes.items():
        body[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
    return json.dumps(body).encode()


# --- the header -------------------------------------------------------------------


def test_every_tensor_counts_the_product_of_its_shape() -> None:
    """[151936, 896] + [896] + a scalar: 136,134,656 + 896 + 1."""
    header = _header(embed=[151936, 896], norm=[896], scale=[])

    assert count_parameters(header, where=WHERE) == 136_135_553


def test_metadata_is_not_a_tensor() -> None:
    assert count_parameters(_header(w=[2, 3]), where=WHERE) == 6


def test_a_zero_dimension_is_an_empty_tensor() -> None:
    assert count_parameters(_header(w=[0, 4096]), where=WHERE) == 0


@pytest.mark.parametrize(
    "header",
    [
        b"\xff\xfe",
        b"null",
        b'{"w": "F16"}',
        b'{"w": {"shape": "2x3"}}',
        b'{"w": {"shape": [2, 3.0]}}',
        b'{"w": {"shape": [-2, 3]}}',
        b'{"w": {"shape": [false, 3]}}',
    ],
)
def test_a_header_that_is_not_one_is_the_miners_to_fix(header: bytes) -> None:
    with pytest.raises(ManifestViolationError):
        count_parameters(header, where=WHERE)


def test_the_declared_length_is_read_little_endian() -> None:
    assert header_length((300).to_bytes(8, "little"), file_size=10_000, where=WHERE) == 300


@pytest.mark.parametrize(
    ("declared", "file_size", "problem"),
    [
        (0, 10_000, "empty"),
        (MAX_HEADER_BYTES + 1, 1 << 40, "over the"),
        (9_993, 10_000, "in a 10000-byte file"),
    ],
)
def test_a_length_the_file_cannot_hold_is_refused_before_it_is_requested(
    declared: int, file_size: int, problem: str
) -> None:
    with pytest.raises(ManifestViolationError, match=problem):
        header_length(declared.to_bytes(8, "little"), file_size=file_size, where=WHERE)


def test_a_file_shorter_than_the_length_prefix_is_refused() -> None:
    with pytest.raises(ManifestViolationError, match="too short"):
        header_length(b"\x01\x00", file_size=2, where=WHERE)


# --- reading it by range ------------------------------------------------------------


FILE = len(_header(w=[1000, 1000])).to_bytes(8, "little") + _header(w=[1000, 1000])


def _snapshot(name: str = "model.safetensors", size: int = 10**9) -> RepoSnapshot:
    return RepoSnapshot(
        repo="org/name",
        revision="a" * 40,
        files=(name,),
        entries=(RepoFile(name, size, "b" * 64),),
    )


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> HuggingFaceClient:
    return HuggingFaceClient(
        endpoint="https://hf.invalid",
        transport=httpx.MockTransport(handler),
        retries=1,
        sleep=lambda _seconds: None,
    )


def _ranged(request: httpx.Request) -> httpx.Response:
    match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.headers["range"])
    assert match is not None
    return httpx.Response(206, content=FILE[int(match.group(1)) : int(match.group(2)) + 1])


def test_two_range_requests_read_the_count() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["range"])
        return _ranged(request)

    with _client(handler) as client:
        assert client.parameter_count(_snapshot()) == 1_000_000

    assert seen == ["bytes=0-7", f"bytes=8-{8 + len(FILE) - 9}"]


def test_a_server_that_ignores_range_is_read_only_as_far_as_needed() -> None:
    """A 200 with the whole file: the prefix is taken and the rest never read."""
    with _client(lambda _request: httpx.Response(200, content=FILE + b"\0" * 4096)) as client:
        assert client.parameter_count(_snapshot()) == 1_000_000


def test_a_body_that_ends_early_is_retried_then_reported_as_an_outage() -> None:
    """The size was checked first, so a short body is the transfer's fault."""
    calls: list[int] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(206, content=b"\x10\x00")

    with _client(handler) as client, pytest.raises(RegistryUnavailableError, match="ended after"):
        client.parameter_count(_snapshot())
    assert len(calls) == 2


def test_a_weight_file_below_the_top_level_cannot_be_counted() -> None:
    with (
        _client(_ranged) as client,
        pytest.raises(ManifestViolationError, match="top of the repository"),
    ):
        client.parameter_count(_snapshot(name="model-x/weights.safetensors"))


def test_a_weight_file_too_small_to_hold_a_header_is_refused_without_a_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing should be requested")

    with _client(handler) as client, pytest.raises(ManifestViolationError, match="too short"):
        client.parameter_count(_snapshot(size=4))
