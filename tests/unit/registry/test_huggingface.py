"""Hugging Face verification: the manifest and the engine hash."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from prometheon.canonical.integrity import canonical_engine_sha256
from prometheon.errors import (
    EngineHashMismatchError,
    ManifestViolationError,
    RegistryError,
    RevisionFormatError,
)
from prometheon.registry.huggingface import (
    MAX_ENGINE_BYTES,
    HuggingFaceClient,
    RepoSnapshot,
    file_allowed,
    file_manifest,
    is_valid_repo_id,
)

from .conftest import (
    CONFORMING_FILES,
    OTHER_REVISION,
    REPO,
    REVISION,
    Calls,
    Handler,
    engine_bytes,
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


def test_manifest_is_read_from_the_wrapper_template() -> None:
    manifest = file_manifest()
    assert "miner.py" in manifest
    assert "chute_config.yml" in manifest
    assert "config.json" in manifest


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


def test_a_repository_without_the_engine_is_a_manifest_violation() -> None:
    files = tuple(name for name in CONFORMING_FILES if name != "miner.py")
    with pytest.raises(ManifestViolationError, match=r"missing miner\.py"):
        client(hf_handler(files=files)).verify_repository(REPO, REVISION)


def test_the_violation_message_is_truncated_for_a_flood_of_files() -> None:
    files = (*CONFORMING_FILES, *(f"junk{index}.py" for index in range(25)))
    with pytest.raises(ManifestViolationError, match="and 15 more"):
        client(hf_handler(files=files)).verify_repository(REPO, REVISION)


# -- engine hash -------------------------------------------------------------


def test_the_canonical_engine_verifies_to_its_own_digest() -> None:
    digest = client(hf_handler()).verify_engine(REPO, REVISION)
    assert digest == canonical_engine_sha256()


def test_one_changed_byte_in_the_engine_is_rejected() -> None:
    tampered = engine_bytes().replace(b"ANSWER:", b"ANSWER!", 1)
    assert tampered != engine_bytes()

    with pytest.raises(EngineHashMismatchError, match="not the canonical engine"):
        client(hf_handler(engine=tampered)).verify_engine(REPO, REVISION)


def test_trailing_whitespace_in_the_engine_is_rejected() -> None:
    with pytest.raises(EngineHashMismatchError):
        client(hf_handler(engine=engine_bytes() + b"\n")).verify_engine(REPO, REVISION)


def test_an_oversized_engine_is_refused_rather_than_buffered() -> None:
    oversized = b"x" * (MAX_ENGINE_BYTES + 1)
    with pytest.raises(RegistryError, match="byte ceiling"):
        client(hf_handler(engine=oversized)).verify_engine(REPO, REVISION)


def test_the_engine_is_fetched_from_the_committed_revision() -> None:
    calls = Calls()
    client(hf_handler(calls=calls)).verify_engine(REPO, REVISION)

    assert calls.paths() == [f"/{REPO}/resolve/{REVISION}/miner.py"]


# -- transport behaviour -----------------------------------------------------


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
