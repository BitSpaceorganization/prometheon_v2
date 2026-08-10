"""Hugging Face verification: the manifest and the engine hash."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from prometheon.errors import (
    ManifestViolationError,
    RegistryError,
    RevisionFormatError,
)
from prometheon.registry.huggingface import (
    HuggingFaceClient,
    RepoSnapshot,
    file_allowed,
    is_valid_repo_id,
)

from .conftest import (
    CONFORMING_FILES,
    OTHER_REVISION,
    REPO,
    REVISION,
    Calls,
    Handler,
    hf_handler,
)

pytestmark = pytest.mark.unit

ENDPOINT = "https://hf.invalid"


def client(
    handler: Handler,
    *,
    token: str | None = None,
    retries: int = 0,
    backoff_seconds: float = 0.0,
    sleep: Callable[[float], None] = lambda _seconds: None,
) -> HuggingFaceClient:
    return HuggingFaceClient(
        endpoint=ENDPOINT,
        token=token,
        transport=httpx.MockTransport(handler),
        retries=retries,
        backoff_seconds=backoff_seconds,
        sleep=sleep,
    )


# -- the manifest comes from the canonical wrapper ---------------------------


def test_sharded_weights_are_allowed_but_arbitrary_files_are_not() -> None:
    assert file_allowed("model-00001-of-00002.safetensors")
    assert file_allowed("model.safetensors")
    assert not file_allowed("backdoor.py")
    assert not file_allowed("model-00001-of-00002.bin")
    # A nested path is not in the manifest, which is what stops a repo hiding
    # code the hash checks never look at.
    assert not file_allowed("scripts/model.safetensors")


@pytest.mark.parametrize(
    "repo",
    ["prometheon-labs/moderation-guard", "bert-base-uncased", "a/b"],
)
def test_repo_ids_the_chain_can_carry_are_accepted(repo: str) -> None:
    assert is_valid_repo_id(repo)


@pytest.mark.parametrize(
    "repo",
    ["", "/name", "owner/", "owner/../etc", "owner/name?x=1", "own er/name", "a/b/c"],
)
def test_repo_ids_that_could_escape_the_url_path_are_rejected(repo: str) -> None:
    assert not is_valid_repo_id(repo)


# -- listing -----------------------------------------------------------------


def test_list_files_returns_the_revision_file_list() -> None:
    calls = Calls()
    snapshot = client(hf_handler(calls=calls)).list_files(REPO, REVISION)

    assert snapshot == RepoSnapshot(repo=REPO, revision=REVISION, files=tuple(CONFORMING_FILES))
    assert calls.paths() == [f"/api/models/{REPO}/revision/{REVISION}"]


def test_a_read_token_is_sent_as_a_bearer_header() -> None:
    calls = Calls()
    client(hf_handler(calls=calls), token="hf_readonly").list_files(REPO, REVISION)

    assert calls.requests[0].headers["authorization"] == "Bearer hf_readonly"


def test_no_authorization_header_without_a_token() -> None:
    calls = Calls()
    client(hf_handler(calls=calls)).list_files(REPO, REVISION)

    assert "authorization" not in calls.requests[0].headers


def test_a_revision_the_api_resolves_elsewhere_is_refused() -> None:
    with pytest.raises(RegistryError, match="resolved to commit"):
        client(hf_handler(sha=OTHER_REVISION)).list_files(REPO, REVISION)


def test_a_branch_name_never_reaches_the_wire() -> None:
    calls = Calls()
    with pytest.raises(RevisionFormatError):
        client(hf_handler(calls=calls)).list_files(REPO, "main")
    assert len(calls) == 0


def test_a_repo_id_that_could_traverse_never_reaches_the_wire() -> None:
    calls = Calls()
    with pytest.raises(RegistryError):
        client(hf_handler(calls=calls)).list_files("owner/../secrets", REVISION)
    assert len(calls) == 0


def test_a_missing_repo_is_a_typed_failure() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    with pytest.raises(RegistryError, match="does not exist"):
        client(handle).list_files(REPO, REVISION)


def test_a_response_without_a_file_list_is_refused() -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"sha": REVISION})

    with pytest.raises(RegistryError, match="no file list"):
        client(handle).list_files(REPO, REVISION)


# -- manifest verification ---------------------------------------------------


def test_a_conforming_repository_passes() -> None:
    hf = client(hf_handler())
    snapshot = hf.verify_repository(REPO, REVISION)
    assert snapshot.files == tuple(CONFORMING_FILES)


def test_an_unlisted_file_is_a_manifest_violation() -> None:
    files = (*CONFORMING_FILES, "backdoor.py")
    with pytest.raises(ManifestViolationError, match=r"backdoor\.py"):
        client(hf_handler(files=files)).verify_repository(REPO, REVISION)


def test_a_nested_file_is_a_manifest_violation() -> None:
    files = (*CONFORMING_FILES, "utils/helper.py")
    with pytest.raises(ManifestViolationError, match=r"utils/helper\.py"):
        client(hf_handler(files=files)).verify_repository(REPO, REVISION)


def test_the_violation_message_is_truncated_for_a_flood_of_files() -> None:
    files = (*CONFORMING_FILES, *(f"junk{index}.py" for index in range(25)))
    with pytest.raises(ManifestViolationError, match="and 15 more"):
        client(hf_handler(files=files)).verify_repository(REPO, REVISION)


# -- engine hash -------------------------------------------------------------


def test_a_transient_server_error_is_retried_then_succeeds(
    no_sleep: Callable[[float], None],
) -> None:
    attempts = {"count": 0}
    inner = hf_handler()

    def handle(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503)
        return inner(request)

    hf = client(handle, retries=2, sleep=no_sleep)
    assert hf.list_files(REPO, REVISION).repo == REPO
    assert attempts["count"] == 2


def test_persistent_unavailability_is_a_typed_failure_not_a_hang(
    no_sleep: Callable[[float], None],
) -> None:
    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(RegistryError, match="unavailable after 3 attempts"):
        client(handle, retries=2, sleep=no_sleep).list_files(REPO, REVISION)


def test_a_connection_failure_is_a_typed_failure(no_sleep: Callable[[float], None]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(RegistryError, match="unavailable"):
        client(handle, retries=1, sleep=no_sleep).list_files(REPO, REVISION)


def test_a_client_error_is_not_retried(no_sleep: Callable[[float], None]) -> None:
    attempts = {"count": 0}

    def handle(_request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(404)

    with pytest.raises(RegistryError):
        client(handle, retries=2, sleep=no_sleep).list_files(REPO, REVISION)
    assert attempts["count"] == 1


def test_backoff_grows_between_attempts() -> None:
    waits: list[float] = []

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    with pytest.raises(RegistryError):
        client(handle, retries=2, backoff_seconds=0.5, sleep=waits.append).list_files(
            REPO, REVISION
        )
    assert waits == [0.5, 1.0]


def test_the_manifest_admits_model_files_and_nothing_executable() -> None:
    """A repository is only ever read for weights now.

    The engine used to be published here and hash-checked, which made this list
    a security boundary: it had to enumerate the ways code could hide, and
    `chute_config.yml` — permitted by name, unchecked in content, applied to the
    image builder — turned out to be one of them. With the engine in the deploy
    script the list becomes a description instead of a defence.
    """
    assert file_allowed("config.json")
    assert file_allowed("model.safetensors")
    assert file_allowed("model-00001-of-00003.safetensors")
    assert file_allowed("tokenizer_config.json")

    for smuggled in ("miner.py", "chute_config.yml", "setup.py", "run.sh", "__init__.py"):
        assert not file_allowed(smuggled), smuggled


def test_a_repository_with_no_weights_is_a_manifest_violation() -> None:
    """Permitted names alone are not a model. There has to be something to serve."""
    with pytest.raises(ManifestViolationError, match="safetensors"):
        client(hf_handler(files=("config.json", "tokenizer.json"))).verify_repository(
            REPO, REVISION
        )
