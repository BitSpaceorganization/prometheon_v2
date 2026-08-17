"""Chutes lookups: hot status, deployed source, and endpoint addressing."""

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
from prometheon.errors import ChuteNotRunningError, RegistryError
from prometheon.registry.chutes import (
    MODERATE_PATH,
    ChuteInfo,
    ChutesClient,
    is_valid_chute_id,
    is_valid_slug,
    require_hot,
)

from .conftest import CHUTE_ID, SLUG, Calls, Handler, chutes_handler, wrapper_source

pytestmark = pytest.mark.unit

API_BASE = "https://chutes-api.invalid"


def client(
    handler: Handler,
    *,
    api_key: str | None = None,
    retries: int = 0,
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> ChutesClient:
    return ChutesClient(
        api_base=API_BASE,
        api_key=api_key,
        transport=httpx.MockTransport(handler),
        retries=retries,
        sleep=sleep,
    )


# -- chute info --------------------------------------------------------------


def test_fetch_chute_reads_slug_and_hot_status() -> None:
    calls = Calls()
    info = client(chutes_handler(calls=calls)).fetch_chute(CHUTE_ID)

    assert info == ChuteInfo(
        chute_id=CHUTE_ID,
        slug=SLUG,
        hot=True,
        name="moderation-guard",
        code="",
        # Read from the nested `image` object the live API returns, owner
        # included: an image id alone says nothing about who built the thing
        # behind it, and the id is only a hash of three strings.
        image_id=image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG),
        image_username=IMAGE_USERNAME,
    )
    assert calls.paths() == [f"/chutes/{CHUTE_ID}"]


def test_the_api_key_is_sent_as_a_bearer_header() -> None:
    calls = Calls()
    client(chutes_handler(calls=calls), api_key="cpk_validator").fetch_chute(CHUTE_ID)

    assert calls.requests[0].headers["authorization"] == "Bearer cpk_validator"


def test_a_cold_chute_reports_cold() -> None:
    assert client(chutes_handler(hot=False)).fetch_chute(CHUTE_ID).hot is False


def test_hot_status_falls_back_to_a_capacity_count() -> None:
    handler = chutes_handler(extra_info={"hot": None, "hot_count": 2})
    assert client(handler).fetch_chute(CHUTE_ID).hot is True

    handler = chutes_handler(extra_info={"hot": None, "hot_count": 0})
    assert client(handler).fetch_chute(CHUTE_ID).hot is False


def test_hot_status_falls_back_to_an_instance_list() -> None:
    handler = chutes_handler(extra_info={"hot": None, "instances": [{"id": "i1"}]})
    assert client(handler).fetch_chute(CHUTE_ID).hot is True


def test_an_unreadable_hot_status_is_refused_rather_than_guessed() -> None:
    handler = chutes_handler(extra_info={"hot": None})
    with pytest.raises(RegistryError, match="did not report whether it is hot"):
        client(handler).fetch_chute(CHUTE_ID)


def test_a_response_describing_a_different_chute_is_refused() -> None:
    handler = chutes_handler(extra_info={"chute_id": "some-other-chute"})
    with pytest.raises(RegistryError, match="different chute"):
        client(handler).fetch_chute(CHUTE_ID)


def test_a_chute_without_a_slug_has_no_endpoint() -> None:
    handler = chutes_handler(slug="")
    with pytest.raises(RegistryError, match="no slug"):
        client(handler).fetch_chute(CHUTE_ID)


def test_a_missing_chute_is_a_typed_failure() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    with pytest.raises(RegistryError, match="does not exist"):
        client(handle).fetch_chute("chute-that-never-existed")


@pytest.mark.parametrize("chute_id", ["", ".", "..", "a/b", "a?b", "x" * 65])
def test_unusable_chute_ids_never_reach_the_wire(chute_id: str) -> None:
    calls = Calls()
    assert not is_valid_chute_id(chute_id)
    with pytest.raises(RegistryError):
        client(chutes_handler(calls=calls)).fetch_chute(chute_id)
    assert len(calls) == 0


# -- deployed source ---------------------------------------------------------


def test_deployed_source_comes_from_the_code_endpoint() -> None:
    calls = Calls()
    chutes = client(chutes_handler(calls=calls))
    info = chutes.fetch_chute(CHUTE_ID)

    assert chutes.deployed_source(info) == wrapper_source()
    assert calls.paths() == [f"/chutes/{CHUTE_ID}", f"/chutes/code/{CHUTE_ID}"]


def test_inline_source_avoids_a_second_round_trip() -> None:
    calls = Calls()
    chutes = client(chutes_handler(inline_code=True, calls=calls))
    info = chutes.fetch_chute(CHUTE_ID)

    assert chutes.deployed_source(info) == wrapper_source()
    assert calls.paths() == [f"/chutes/{CHUTE_ID}"]


def test_a_json_wrapped_source_is_unwrapped() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/chutes/code/{CHUTE_ID}":
            return httpx.Response(200, json={"code": wrapper_source()})
        return chutes_handler()(request)

    assert client(handle).fetch_deployed_source(CHUTE_ID) == wrapper_source()


def test_a_json_object_with_no_source_is_an_upstream_failure_not_a_miner_failure() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "pending"})

    with pytest.raises(RegistryError, match="carrying no source"):
        client(handle).fetch_deployed_source(CHUTE_ID)


def test_an_empty_source_is_refused() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"   \n")

    with pytest.raises(RegistryError, match="is empty"):
        client(handle).fetch_deployed_source(CHUTE_ID)


def test_a_non_utf8_source_is_refused() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\xff\xfe not text")

    with pytest.raises(RegistryError, match="not UTF-8"):
        client(handle).fetch_deployed_source(CHUTE_ID)


# -- addressing --------------------------------------------------------------


def test_the_endpoint_is_built_from_the_slug() -> None:
    chutes = client(chutes_handler())
    assert chutes.endpoint_url(SLUG) == f"https://{SLUG}.chutes.ai"
    assert chutes.moderate_url(SLUG) == f"https://{SLUG}.chutes.ai{MODERATE_PATH}"


@pytest.mark.parametrize(
    "slug",
    [
        "",
        "evil.attacker.example",
        "slug/../other",
        "SLUG",
        "slug_with_underscore",
        "-leading-dash",
        "trailing-dash-",
        "a" * 64,
    ],
)
def test_a_slug_that_is_not_a_dns_label_cannot_become_a_hostname(slug: str) -> None:
    assert not is_valid_slug(slug)
    with pytest.raises(RegistryError, match="chute slug"):
        client(chutes_handler()).endpoint_url(slug)


def test_an_endpoint_template_without_a_slug_placeholder_is_refused() -> None:
    with pytest.raises(RegistryError, match=r"\{slug\}"):
        ChutesClient(
            api_base=API_BASE,
            endpoint_template="https://fixed.chutes.ai",
            transport=httpx.MockTransport(chutes_handler()),
        )


def test_require_hot_raises_the_registry_error_a_scorer_can_key_on() -> None:
    cold = ChuteInfo(chute_id=CHUTE_ID, slug=SLUG, hot=False)
    with pytest.raises(ChuteNotRunningError) as caught:
        require_hot(cold)
    assert caught.value.code == "registry.chute_not_running"

    hot = ChuteInfo(chute_id=CHUTE_ID, slug=SLUG, hot=True)
    assert require_hot(hot) is hot


@pytest.mark.parametrize(
    "slug",
    [
        "guard-8b",  # no prefix at all
        "prometheon",  # the prefix alone addresses nothing
        "prometheon-",  # separator with no name after it
        "prometheonguard",  # prefix must be a whole label, not a substring
        "other-prometheon-x",  # prefix must lead, not appear anywhere
        "PROMETHEON-guard",  # a hostname label is lowercase
    ],
)
def test_a_chute_outside_the_namespace_is_refused(slug: str) -> None:
    """The prefix is what makes a deployment provably ours.

    Without it a miner could commit any chute on the platform — including
    someone else's working model — and be scored on a deployment they do not
    control.
    """
    chutes = client(chutes_handler())
    with pytest.raises(RegistryError):
        chutes.endpoint_url(slug)


def test_a_namespaced_slug_is_accepted() -> None:
    chutes = client(chutes_handler())
    assert chutes.endpoint_url("prometheon-guard-8b") == "https://prometheon-guard-8b.chutes.ai"
