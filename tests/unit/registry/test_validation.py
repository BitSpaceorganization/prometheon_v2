"""The eligibility pipeline, end to end, over a mocked Hugging Face."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest

from prometheon.chain.commitment import ModelCommitment
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.validation import (
    InvalidReason,
    MinerEntry,
    MinerValidation,
    ModelRegistry,
    eligible_miners,
)

from .conftest import CONFORMING_FILES, Calls

pytestmark = pytest.mark.unit

HF_ENDPOINT = "https://hf.invalid"

REPO_A = "miner-one/guard"
REPO_B = "miner-two/guard"
REV_A = "a" * 40
REV_B = "b" * 40

HOTKEY_A = "5FA0miner_one"
HOTKEY_B = "5FA0miner_two"

#: Comfortably inside the ceiling the fixtures set.
SHARD_BYTES = 4 * 1024**3
CEILING = 24 * 1024**3


@dataclass
class Repo:
    """One Hugging Face repository at one revision, as the API would serve it."""

    files: tuple[str, ...] = CONFORMING_FILES
    #: What `config.json` declares. `qwen2` is a real architecture the pinned
    #: transformers knows, so the default exercises the accepting path.
    model_type: str = "qwen2"
    shard_bytes: int = SHARD_BYTES


class World:
    """A fixed Hugging Face universe the registry reads from."""

    def __init__(
        self,
        *,
        repos: dict[tuple[str, str], Repo] | None = None,
        hf_down: bool = False,
        max_weight_bytes: int = CEILING,
    ) -> None:
        self.repos = repos if repos is not None else {(REPO_A, REV_A): Repo()}
        self.hf_down = hf_down
        self.max_weight_bytes = max_weight_bytes
        self.hf_calls = Calls()

    def registry(self) -> ModelRegistry:
        return ModelRegistry(
            huggingface=HuggingFaceClient(
                endpoint=HF_ENDPOINT,
                transport=httpx.MockTransport(self._hf),
                retries=0,
            ),
            max_weight_bytes=self.max_weight_bytes,
        )

    def _hf(self, request: httpx.Request) -> httpx.Response:
        self.hf_calls.requests.append(request)
        if self.hf_down:
            return httpx.Response(503)
        path = request.url.path
        if path.startswith("/api/models/") and "/revision/" in path:
            repo, _, revision = path[len("/api/models/") :].partition("/revision/")
            state = self.repos.get((repo, revision))
            if state is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(
                200,
                json={
                    "sha": revision,
                    "siblings": [
                        {
                            "rfilename": name,
                            "size": state.shard_bytes if name.endswith(".safetensors") else 512,
                        }
                        for name in state.files
                    ],
                },
            )
        if "/resolve/" in path:
            repo, _, tail = path.lstrip("/").partition("/resolve/")
            revision, _, _filename = tail.partition("/")
            state = self.repos.get((repo, revision))
            if state is None:
                return httpx.Response(404, json={"error": "not found"})
            body = json.dumps({"model_type": state.model_type} if state.model_type else {})
            return httpx.Response(200, content=body.encode())
        return httpx.Response(404, json={"error": "not found"})


def commitment(*, repo: str = REPO_A, revision: str = REV_A) -> ModelCommitment:
    return ModelCommitment(hf_repo=repo, hf_revision=revision)


def forced(commit: ModelCommitment, **overrides: str) -> ModelCommitment:
    """A commitment carrying values its own constructor would have rejected.

    The registry must reach its own verdict rather than inherit the decoder's
    invariants, so the only way to exercise those checks is to bypass the
    constructor the way a future decoder change might.
    """
    for name, value in overrides.items():
        object.__setattr__(commit, name, value)
    return commit


def entry(
    *,
    uid: int = 1,
    hotkey: str = HOTKEY_A,
    commit: ModelCommitment | None = None,
    decode_error: str | None = None,
    block: int = 100,
) -> MinerEntry:
    return MinerEntry(
        uid=uid,
        hotkey=hotkey,
        commitment=commitment() if commit is None and decode_error is None else commit,
        decode_error=decode_error,
        commit_block=block,
    )


def only(world: World, miner: MinerEntry) -> MinerValidation:
    return world.registry().validate([miner])[0]


# -- the happy path ----------------------------------------------------------


def test_a_conforming_miner_is_eligible() -> None:
    result = only(World(), entry())

    assert result.valid
    assert result.reason is None
    assert result.hf_repo == REPO_A
    assert result.hf_revision == REV_A
    assert result.commit_block == 100
    assert result.model_type == "qwen2"
    assert result.weight_bytes == 2 * SHARD_BYTES


def test_the_verdict_carries_what_evaluation_needs_to_fetch_the_model() -> None:
    """Nothing else is needed: the repo and the SHA *are* the submission."""
    result = only(World(), entry())
    assert (result.hf_repo, result.hf_revision) == (REPO_A, REV_A)


def test_eligible_miners_keeps_only_the_valid_ones() -> None:
    world = World()
    results = world.registry().validate([entry(uid=1), MinerEntry(uid=2, hotkey=HOTKEY_B)])
    assert [r.uid for r in eligible_miners(results)] == [1]


# -- what a miner can get wrong ---------------------------------------------


def test_a_hotkey_that_never_committed_is_reported_as_such() -> None:
    result = only(World(), MinerEntry(uid=3, hotkey=HOTKEY_A))
    assert result.reason is InvalidReason.COMMITMENT_MISSING


def test_an_undecodable_commitment_is_distinguished_from_a_missing_one() -> None:
    result = only(World(), entry(decode_error="unknown format version 'z'"))
    assert result.reason is InvalidReason.COMMITMENT_MALFORMED


def test_a_moving_reference_is_not_a_commitment() -> None:
    result = only(World(), entry(commit=forced(commitment(), hf_revision="main")))
    assert result.reason is InvalidReason.REVISION_NOT_SHA


def test_executable_code_in_the_repository_makes_a_miner_ineligible() -> None:
    world = World(repos={(REPO_A, REV_A): Repo(files=(*CONFORMING_FILES, "run.py"))})
    assert only(world, entry()).reason is InvalidReason.MANIFEST_VIOLATION


def test_a_repository_that_does_not_exist_is_an_upstream_failure() -> None:
    assert only(World(repos={}), entry()).reason is InvalidReason.UPSTREAM_UNAVAILABLE


def test_hugging_face_being_down_never_marks_a_miner_dishonest() -> None:
    assert only(World(hf_down=True), entry()).reason is InvalidReason.UPSTREAM_UNAVAILABLE


# -- what only matters because validators run the model ----------------------


class TestTheModelHasToBeRunnableByEveryValidator:
    """Both checks exist because evaluation moved onto validator hardware.

    A model that cannot load, or cannot fit, is not one miner overreaching on a
    budget they pay for. It is work every validator attempts and fails at, on
    the same day, having already spent the bandwidth.
    """

    def test_an_architecture_the_runtime_cannot_load_is_refused(self) -> None:
        world = World(repos={(REPO_A, REV_A): Repo(model_type="qwen3")})
        result = only(world, entry())

        assert result.reason is InvalidReason.ARCHITECTURE_UNSUPPORTED
        assert "qwen3" in result.detail

    def test_a_checkpoint_declaring_no_architecture_is_not_guessed_at(self) -> None:
        """Silence is not evidence of a bad model."""
        world = World(repos={(REPO_A, REV_A): Repo(model_type="")})
        assert only(world, entry()).valid

    def test_weights_over_the_ceiling_are_refused(self) -> None:
        world = World(repos={(REPO_A, REV_A): Repo(shard_bytes=20 * 1024**3)})
        result = only(world, entry())

        assert result.reason is InvalidReason.MODEL_TOO_LARGE
        assert "GiB" in result.detail

    def test_the_ceiling_is_the_configured_one_not_a_constant(self) -> None:
        """It describes hardware validators agreed to run, so it is policy."""
        world = World(max_weight_bytes=4 * 1024**3)
        assert only(world, entry()).reason is InvalidReason.MODEL_TOO_LARGE

    def test_size_is_read_before_anything_is_downloaded(self) -> None:
        """One API request, not an hour of bandwidth on every validator."""
        world = World(repos={(REPO_A, REV_A): Repo(shard_bytes=20 * 1024**3)})
        only(world, entry())

        assert not any("/resolve/" in p and "safetensors" in p for p in world.hf_calls.paths())


# -- duplicates --------------------------------------------------------------


def _two_miners_on_one_model(*, block_a: int, block_b: int) -> tuple[World, list[MinerEntry]]:
    world = World(repos={(REPO_A, REV_A): Repo(), (REPO_B, REV_A): Repo()})
    entries = [
        entry(uid=1, hotkey=HOTKEY_A, commit=commitment(repo=REPO_A), block=block_a),
        entry(uid=2, hotkey=HOTKEY_B, commit=commitment(repo=REPO_B), block=block_b),
    ]
    return world, entries


class TestMirroringDoesNotDefeatTheDuplicateCheck:
    """Grouped on the revision SHA alone, never on the repo id.

    A Hugging Face repo is a git repo: `git clone --mirror` into another
    namespace preserves every commit SHA byte for byte. A key including the
    repo string put the original and its mirror in different groups and flagged
    neither.
    """

    def test_a_mirrored_repo_with_the_same_sha_is_caught(self) -> None:
        world, entries = _two_miners_on_one_model(block_a=500, block_b=100)
        first, second = world.registry().validate(entries)

        assert first.reason is InvalidReason.DUPLICATE_MODEL
        assert second.valid

    def test_the_earliest_commitment_wins_regardless_of_input_order(self) -> None:
        world, entries = _two_miners_on_one_model(block_a=100, block_b=500)
        first, second = world.registry().validate(entries)

        assert first.valid
        assert second.reason is InvalidReason.DUPLICATE_MODEL

    def test_genuinely_distinct_models_are_not_grouped(self) -> None:
        world = World(repos={(REPO_A, REV_A): Repo(), (REPO_B, REV_B): Repo()})
        results = world.registry().validate(
            [
                entry(uid=1, hotkey=HOTKEY_A, commit=commitment(repo=REPO_A, revision=REV_A)),
                entry(uid=2, hotkey=HOTKEY_B, commit=commitment(repo=REPO_B, revision=REV_B)),
            ]
        )
        assert all(result.valid for result in results)


def test_results_are_returned_in_the_order_given() -> None:
    world = World()
    results = world.registry().validate([entry(uid=7), entry(uid=3, hotkey=HOTKEY_B), entry(uid=5)])
    assert [result.uid for result in results] == [7, 3, 5]
