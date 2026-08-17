"""The eligibility pipeline, end to end, over mocked Hugging Face and Chutes."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx
import pytest

from prometheon.canonical.hashes import IMAGE_NAME, IMAGE_TAG, IMAGE_USERNAME, image_id_for
from prometheon.chain.commitment import ModelCommitment
from prometheon.registry.chutes import ChutesClient
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.validation import (
    InvalidReason,
    MinerEntry,
    MinerValidation,
    ModelRegistry,
    eligible_miners,
)

from .conftest import CONFORMING_FILES, Calls, wrapper_source

pytestmark = pytest.mark.unit

HF_ENDPOINT = "https://hf.invalid"
CHUTES_API = "https://chutes-api.invalid"

REPO_A = "miner-one/guard"
REPO_B = "miner-two/guard"
REV_A = "a" * 40
REV_B = "b" * 40

CHUTE_A = "11111111-1111-4111-8111-111111111111"
CHUTE_B = "22222222-2222-4222-8222-222222222222"

HOTKEY_A = "5FA0miner_one"
HOTKEY_B = "5FA0miner_two"


@dataclass
class Repo:
    """One Hugging Face repository at one revision, as the API would serve it."""

    files: tuple[str, ...] = CONFORMING_FILES


@dataclass
class Deployment:
    """One chute, and the source the platform holds for it."""

    slug: str
    hot: bool = True
    #: What the deployed wrapper pins. Defaults to a self-consistent wrapper.
    declares_repo: str = REPO_A
    declares_revision: str = REV_A
    #: Overrides the rendered wrapper entirely, for tampering tests.
    source: str | None = None
    #: Whatever the platform's own metadata claims, which is never trusted.
    metadata: dict[str, object] = field(default_factory=dict)
    #: The image the platform says this chute runs. Defaults to the subnet's.
    image_id: str | None = None
    image_username: str = IMAGE_USERNAME

    def deployed_source(self) -> str:
        if self.source is not None:
            return self.source
        return wrapper_source(
            hf_repo=self.declares_repo,
            hf_revision=self.declares_revision,
        )


class World:
    """A fixed Hugging Face and Chutes universe the registry reads from."""

    def __init__(
        self,
        *,
        repos: dict[tuple[str, str], Repo] | None = None,
        chutes: dict[str, Deployment] | None = None,
        hf_down: bool = False,
    ) -> None:
        self.repos = repos if repos is not None else {(REPO_A, REV_A): Repo()}
        self.chutes = (
            chutes if chutes is not None else {CHUTE_A: Deployment(slug="prometheon-guard-a")}
        )
        self.hf_down = hf_down
        self.hf_calls = Calls()
        self.chute_calls = Calls()

    def registry(self) -> ModelRegistry:
        return ModelRegistry(
            huggingface=HuggingFaceClient(
                endpoint=HF_ENDPOINT,
                transport=httpx.MockTransport(self._hf),
                retries=0,
            ),
            chutes=ChutesClient(
                api_base=CHUTES_API,
                transport=httpx.MockTransport(self._chutes),
                retries=0,
            ),
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
                    "siblings": [{"rfilename": name} for name in state.files],
                },
            )
        if "/resolve/" in path:
            repo, _, tail = path.lstrip("/").partition("/resolve/")
            revision, _, _filename = tail.partition("/")
            state = self.repos.get((repo, revision))
            if state is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(200, content=state.engine)
        return httpx.Response(404, json={"error": "not found"})

    def _chutes(self, request: httpx.Request) -> httpx.Response:
        self.chute_calls.requests.append(request)
        path = request.url.path
        if path.startswith("/chutes/code/"):
            deployment = self.chutes.get(path[len("/chutes/code/") :])
            if deployment is None:
                return httpx.Response(404, json={"error": "not found"})
            return httpx.Response(200, content=deployment.deployed_source().encode("utf-8"))
        if path.startswith("/chutes/"):
            chute_id = path[len("/chutes/") :]
            deployment = self.chutes.get(chute_id)
            if deployment is None:
                return httpx.Response(404, json={"error": "not found"})
            info: dict[str, object] = {
                "chute_id": chute_id,
                "slug": deployment.slug,
                "hot": deployment.hot,
                "image": {
                    "image_id": (
                        deployment.image_id
                        if deployment.image_id is not None
                        else image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG)
                    ),
                    "user": {"username": deployment.image_username},
                },
            }
            info.update(deployment.metadata)
            return httpx.Response(200, json=info)
        return httpx.Response(404, json={"error": "not found"})


def commitment(
    *, repo: str = REPO_A, revision: str = REV_A, chute_id: str = CHUTE_A
) -> ModelCommitment:
    return ModelCommitment(hf_repo=repo, hf_revision=revision, chute_id=chute_id)


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


def test_a_conforming_miner_is_eligible_and_addressable() -> None:
    result = only(World(), entry())

    assert result.valid
    assert result.reason is None
    assert result.hf_repo == REPO_A
    assert result.hf_revision == REV_A
    assert result.chute_id == CHUTE_A
    assert result.commit_block == 100
    assert result.chute_slug == "prometheon-guard-a"
    assert result.endpoint_url == "https://prometheon-guard-a.chutes.ai"
    assert result.moderate_url == "https://prometheon-guard-a.chutes.ai/moderate"
    assert result.wrapper_digest


def test_an_ineligible_miner_carries_no_endpoint() -> None:
    result = only(World(), MinerEntry(uid=7, hotkey=HOTKEY_B))

    assert not result.valid
    assert result.endpoint_url == ""
    assert result.moderate_url == ""


def test_eligible_miners_keeps_only_the_valid_ones() -> None:
    world = World(
        repos={(REPO_A, REV_A): Repo()},
        chutes={CHUTE_A: Deployment(slug="prometheon-guard-a")},
    )
    results = world.registry().validate([entry(uid=1), MinerEntry(uid=2, hotkey=HOTKEY_B)])
    assert [result.uid for result in eligible_miners(results)] == [1]


# -- step 1: the commitment --------------------------------------------------


def test_a_hotkey_that_never_committed_is_reported_as_such() -> None:
    result = only(World(), MinerEntry(uid=3, hotkey=HOTKEY_A))
    assert result.reason is InvalidReason.COMMITMENT_MISSING


def test_an_undecodable_commitment_is_distinguished_from_a_missing_one() -> None:
    result = only(World(), entry(decode_error="unknown chute_id encoding mode 'z'"))
    assert result.reason is InvalidReason.COMMITMENT_MALFORMED
    assert "encoding mode" in result.detail


def test_a_repo_id_that_could_escape_the_url_is_treated_as_malformed() -> None:
    world = World()
    result = only(world, entry(commit=forced(commitment(), hf_repo="owner/../secrets")))

    assert result.reason is InvalidReason.COMMITMENT_MALFORMED
    assert len(world.hf_calls) == 0


def test_no_request_is_made_for_a_miner_without_a_commitment() -> None:
    world = World()
    world.registry().validate([MinerEntry(uid=1, hotkey=HOTKEY_A)])
    assert len(world.hf_calls) == 0
    assert len(world.chute_calls) == 0


# -- step 2: the revision ----------------------------------------------------


def test_a_moving_reference_is_not_a_commitment() -> None:
    world = World()
    result = only(world, entry(commit=forced(commitment(), hf_revision="main")))

    assert result.reason is InvalidReason.REVISION_NOT_SHA
    assert len(world.hf_calls) == 0


# -- step 3: the published repository ----------------------------------------


def test_an_unlisted_file_makes_a_miner_ineligible() -> None:
    world = World(repos={(REPO_A, REV_A): Repo(files=(*CONFORMING_FILES, "backdoor.py"))})
    assert only(world, entry()).reason is InvalidReason.MANIFEST_VIOLATION


def test_executable_code_in_the_repository_makes_a_miner_ineligible() -> None:
    """The engine used to live here and be hashed. Now nothing executable does.

    A repository is only ever read for weights, so the manifest stops being a
    list of the ways code can hide and becomes a description of model files.
    `chute_config.yml` was the file that made the old version of this necessary:
    permitted by name, unchecked in content, and applied to the image builder.
    """
    for smuggled in ("miner.py", "chute_config.yml", "setup.py"):
        world = World(repos={(REPO_A, REV_A): Repo(files=(*CONFORMING_FILES, smuggled))})
        assert only(world, entry()).reason is InvalidReason.MANIFEST_VIOLATION, smuggled


def test_a_deployment_running_another_image_makes_a_miner_ineligible() -> None:
    """The check that makes "every model runs identical code" true.

    Everything else proves which bytes are deployed. This is what makes them run
    somewhere the miner did not assemble — and it is the only question about a
    runtime the platform can answer, because the API exposes an image's identity
    and never its contents.
    """
    world = World(
        chutes={
            CHUTE_A: Deployment(
                slug="prometheon-guard-a", image_id="00000000-0000-0000-0000-000000000000"
            )
        }
    )
    assert only(world, entry()).reason is InvalidReason.IMAGE_MISMATCH


def test_an_image_of_the_right_name_owned_by_someone_else_is_refused() -> None:
    """An id is a hash of three strings the miner picks.

    Anyone can build `their-account/prometheon-moderation:1` from any Dockerfile
    at all. Only the owner distinguishes ours from theirs.
    """
    world = World(
        chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", image_username="somebody-else")}
    )
    assert only(world, entry()).reason is InvalidReason.IMAGE_MISMATCH


def test_a_repository_that_does_not_exist_is_an_upstream_failure() -> None:
    world = World(repos={})
    result = only(world, entry())
    assert result.reason is InvalidReason.UPSTREAM_UNAVAILABLE


def test_hugging_face_being_down_never_marks_a_miner_dishonest() -> None:
    world = World(hf_down=True)
    result = only(world, entry())

    assert result.reason is InvalidReason.UPSTREAM_UNAVAILABLE
    assert result.reason.value == "registry.error"


# -- step 4: the deployed wrapper --------------------------------------------


def test_an_edited_wrapper_is_rejected() -> None:
    # A one-word change to the verdict rule: ties resolve to YES instead of NO.
    # Behaviour-changing, invisible in a diff of the deployment's metadata, and
    # exactly what the hash exists to catch.
    original = wrapper_source(hf_repo=REPO_A, hf_revision=REV_A)
    tampered = original.replace("return bool(yes > no)", "return bool(yes >= no)")
    assert tampered != original

    world = World(chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", source=tampered)})

    assert only(world, entry()).reason is InvalidReason.WRAPPER_HASH_MISMATCH


def test_a_reformatted_wrapper_is_still_the_canonical_wrapper() -> None:
    # Comments and blank lines are outside the AST, so a miner cannot invalidate
    # their own deployment by reformatting it.
    reformatted = (
        "# a note the miner added\n\n"
        + wrapper_source(hf_repo=REPO_A, hf_revision=REV_A)
        + "\n\n# and a trailing one\n"
    )
    world = World(chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", source=reformatted)})

    assert only(world, entry()).valid


def test_source_that_is_not_python_is_rejected_as_a_wrapper_mismatch() -> None:
    world = World(chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", source="def broken(:\n")})
    result = only(world, entry())

    assert result.reason is InvalidReason.WRAPPER_HASH_MISMATCH
    assert "not parseable" in result.detail


# -- step 5: deployed source vs the chain ------------------------------------


def test_a_wrapper_serving_another_revision_is_rejected() -> None:
    world = World(
        repos={(REPO_A, REV_A): Repo(), (REPO_A, REV_B): Repo()},
        chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", declares_revision=REV_B)},
    )
    result = only(world, entry())

    assert result.reason is InvalidReason.COMMITMENT_MISMATCH
    assert REV_B in result.detail


def test_a_wrapper_serving_another_repo_is_rejected() -> None:
    world = World(
        repos={(REPO_A, REV_A): Repo()},
        chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", declares_repo=REPO_B)},
    )
    assert only(world, entry()).reason is InvalidReason.COMMITMENT_MISMATCH


def test_chute_metadata_cannot_rescue_a_wrapper_that_pins_a_branch() -> None:
    # Why the deployed source is read at all: metadata claiming the committed
    # revision means nothing, because a miner controls it and it is not covered
    # by the wrapper hash.
    branch_pinned = wrapper_source(hf_repo=REPO_A, hf_revision=REV_A).replace(
        f'PROMETHEON_HF_REVISION = "{REV_A}"',
        'PROMETHEON_HF_REVISION = "main"',
    )
    world = World(
        chutes={
            CHUTE_A: Deployment(
                slug="prometheon-guard-a",
                source=branch_pinned,
                metadata={"hf_revision": REV_A, "revision": REV_A},
            )
        }
    )
    result = only(world, entry())

    assert result.reason is InvalidReason.COMMITMENT_MISMATCH
    assert "'main'" in result.detail


def test_the_chute_id_in_the_wrapper_is_not_compared_with_the_commitment() -> None:
    # PROMETHEON_CHUTE_ID is the deployment *name*; the platform assigns the id
    # the chain carries. Comparing them would invalidate every honest miner.
    world = World(chutes={CHUTE_A: Deployment(slug="prometheon-guard-a")})
    assert only(world, entry()).valid


# -- step 6: the deployment is live ------------------------------------------


def test_a_cold_chute_is_not_scorable() -> None:
    world = World(chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", hot=False)})
    result = only(world, entry())

    assert result.reason is InvalidReason.CHUTE_NOT_RUNNING
    assert result.endpoint_url == ""


def test_a_slug_that_cannot_become_a_hostname_is_not_scorable() -> None:
    world = World(chutes={CHUTE_A: Deployment(slug="evil.attacker.example")})
    result = only(world, entry())

    assert result.reason is InvalidReason.CHUTE_NOT_RUNNING


def test_a_chute_that_does_not_exist_is_an_upstream_failure() -> None:
    world = World(chutes={})
    assert only(world, entry()).reason is InvalidReason.UPSTREAM_UNAVAILABLE


# -- step 7: duplicates ------------------------------------------------------


def _two_miners_on_one_model(*, block_a: int, block_b: int) -> tuple[World, list[MinerEntry]]:
    world = World(
        repos={(REPO_A, REV_A): Repo()},
        chutes={
            CHUTE_A: Deployment(slug="prometheon-guard-a"),
            CHUTE_B: Deployment(slug="prometheon-guard-b"),
        },
    )
    entries = [
        entry(uid=1, hotkey=HOTKEY_A, commit=commitment(chute_id=CHUTE_A), block=block_a),
        entry(uid=2, hotkey=HOTKEY_B, commit=commitment(chute_id=CHUTE_B), block=block_b),
    ]
    return world, entries


def test_the_earliest_commitment_of_a_model_wins() -> None:
    world, entries = _two_miners_on_one_model(block_a=500, block_b=100)
    first, second = world.registry().validate(entries)

    assert first.reason is InvalidReason.DUPLICATE_MODEL
    assert f"uid {entries[1].uid}" in first.detail
    assert second.valid


def test_a_same_block_tie_is_broken_by_uid_so_validators_agree() -> None:
    world, entries = _two_miners_on_one_model(block_a=100, block_b=100)
    first, second = world.registry().validate(entries)

    assert first.valid
    assert second.reason is InvalidReason.DUPLICATE_MODEL


def test_a_copy_does_not_become_eligible_when_the_original_is_cold() -> None:
    # Otherwise the cheapest strategy in the subnet is to copy someone else's
    # model and wait for their deployment to blink.
    world = World(
        repos={(REPO_A, REV_A): Repo()},
        chutes={
            CHUTE_A: Deployment(slug="prometheon-guard-a", hot=False),
            CHUTE_B: Deployment(slug="prometheon-guard-b"),
        },
    )
    original = entry(uid=1, hotkey=HOTKEY_A, commit=commitment(chute_id=CHUTE_A), block=100)
    copy = entry(uid=2, hotkey=HOTKEY_B, commit=commitment(chute_id=CHUTE_B), block=900)

    first, second = world.registry().validate([original, copy])

    assert first.reason is InvalidReason.CHUTE_NOT_RUNNING
    assert second.reason is InvalidReason.DUPLICATE_MODEL


def test_different_revisions_of_one_repo_are_not_duplicates() -> None:
    world = World(
        repos={(REPO_A, REV_A): Repo(), (REPO_A, REV_B): Repo()},
        chutes={
            CHUTE_A: Deployment(slug="prometheon-guard-a"),
            CHUTE_B: Deployment(slug="prometheon-guard-b", declares_revision=REV_B),
        },
    )
    entries = [
        entry(uid=1, commit=commitment(chute_id=CHUTE_A), block=100),
        entry(
            uid=2,
            hotkey=HOTKEY_B,
            commit=commitment(revision=REV_B, chute_id=CHUTE_B),
            block=200,
        ),
    ]
    assert all(result.valid for result in world.registry().validate(entries))


# -- ordering and cost -------------------------------------------------------


def test_the_first_failure_in_pipeline_order_is_the_one_reported() -> None:
    # This miner fails the manifest *and* is cold. The manifest is what they
    # need to fix first, so that is what they are told.
    world = World(
        repos={(REPO_A, REV_A): Repo(files=(*CONFORMING_FILES, "backdoor.py"))},
        chutes={CHUTE_A: Deployment(slug="prometheon-guard-a", hot=False)},
    )
    result = only(world, entry())

    assert result.reason is InvalidReason.MANIFEST_VIOLATION
    assert len(world.chute_calls) == 0


def test_one_repository_is_verified_once_however_many_miners_claim_it() -> None:
    world, entries = _two_miners_on_one_model(block_a=100, block_b=200)
    world.registry().validate(entries)

    # One request per repository verification: the file list, and nothing else.
    # The engine used to be fetched and hashed from here too; it lives in the
    # deploy script now, so a repository is only ever read for its file names.
    assert len(world.hf_calls) == 1


def test_a_deterministic_repository_verdict_is_reused_across_miners() -> None:
    world = World(
        repos={(REPO_A, REV_A): Repo(files=(*CONFORMING_FILES, "backdoor.py"))},
        chutes={
            CHUTE_A: Deployment(slug="prometheon-guard-a"),
            CHUTE_B: Deployment(slug="prometheon-guard-b"),
        },
    )
    entries = [
        entry(uid=1, commit=commitment(chute_id=CHUTE_A), block=100),
        entry(uid=2, hotkey=HOTKEY_B, commit=commitment(chute_id=CHUTE_B), block=200),
    ]
    results = world.registry().validate(entries)

    assert [result.reason for result in results] == [
        InvalidReason.MANIFEST_VIOLATION,
        InvalidReason.MANIFEST_VIOLATION,
    ]
    assert len(world.hf_calls) == 1


def test_an_upstream_outage_is_retried_per_miner_rather_than_cached() -> None:
    world = World(hf_down=True)
    entries = [
        entry(uid=1, commit=commitment(chute_id=CHUTE_A), block=100),
        entry(uid=2, hotkey=HOTKEY_B, commit=commitment(chute_id=CHUTE_B), block=200),
    ]
    world.registry().validate(entries)

    assert len(world.hf_calls) == 2


def test_results_are_returned_in_the_order_given() -> None:
    world, entries = _two_miners_on_one_model(block_a=100, block_b=200)
    results = world.registry().validate(entries)
    assert [result.uid for result in results] == [1, 2]


def test_every_reason_carries_the_error_taxonomy_code() -> None:
    assert InvalidReason.DUPLICATE_MODEL.value == "registry.duplicate_model"
    assert InvalidReason.COMMITMENT_MISSING.value == "chain.commitment_error"
    assert InvalidReason.COMMITMENT_MALFORMED.value == "chain.commitment_malformed"
    assert len({reason.value for reason in InvalidReason}) == len(list(InvalidReason))


class TestOneMinerCannotAbortTheCycle:
    """Deployed source is attacker-controlled and must be contained.

    Everything here reaches ``ast`` via a string fetched from a miner's own
    chute. An exception that escapes ``validate()`` does not fail that miner.
    It aborts eligibility for the whole field, on every validator, for the day.
    """

    def test_a_deeply_nested_source_is_a_verdict_not_a_crash(self) -> None:
        """Valid Python, 39 KB, under the size cap, and it blows the C stack.

        ``normalise_source`` walks the tree with ``ast.dump``, which recurses.
        ``RecursionError`` is a ``RuntimeError``, so it was caught by nothing
        on the way out. The attack is free: the miner is ineligible either way.
        """
        from prometheon.errors import WrapperHashMismatchError
        from prometheon.registry.validation import _hash_hostile_source

        hostile = "x = " + "+".join(["1"] * 20_000)
        with pytest.raises(WrapperHashMismatchError, match="nests too deeply"):
            _hash_hostile_source(hostile)

    def test_source_that_is_not_python_at_all_is_a_verdict(self) -> None:
        from prometheon.errors import RegistryError
        from prometheon.registry.validation import _hash_hostile_source

        with pytest.raises(RegistryError):
            _hash_hostile_source("this is not python {{{")

    def test_a_nul_byte_is_a_verdict_on_every_supported_python(self) -> None:
        """3.10 and 3.11 raise ValueError here, not SyntaxError."""
        from prometheon.errors import RegistryError
        from prometheon.registry.validation import _hash_hostile_source

        with pytest.raises(RegistryError):
            _hash_hostile_source("x = 1\x00")


class TestMirroringDoesNotDefeatTheDuplicateCheck:
    """A HF model repo is a git repo, so SHAs survive a mirror.

    ``git clone --mirror`` into another namespace preserves every commit SHA
    byte for byte. The copycat's manifest, engine hash and declared repo are
    all honest, so checks 1-6 pass; only the repo string differs. Keying the
    duplicate group on it let one model hold rank 1 *and* rank 2, taking 4500
    of the 5000bp model pool with a second hotkey.
    """

    def test_a_mirrored_repo_with_the_same_sha_is_caught(self) -> None:
        from prometheon.registry.validation import _duplicate_losers

        original = MinerEntry(
            uid=1,
            hotkey="5Orig",
            commitment=ModelCommitment("orig/guard", REV_A, "chute-1"),
            commit_block=100,
        )
        mirror = MinerEntry(
            uid=2,
            hotkey="5Copy",
            commitment=ModelCommitment("copycat/guard", REV_A, "chute-2"),
            commit_block=900,
        )

        losers = _duplicate_losers([original, mirror])

        assert 1 in losers, "the later mirror must be flagged"
        assert 0 not in losers, "the original must keep its claim"
        assert losers[1].hotkey == "5Orig"

    def test_the_first_committer_wins_regardless_of_input_order(self) -> None:
        from prometheon.registry.validation import _duplicate_losers

        original = MinerEntry(
            uid=9,
            hotkey="5Orig",
            commitment=ModelCommitment("orig/guard", REV_A, "chute-1"),
            commit_block=100,
        )
        mirror = MinerEntry(
            uid=2,
            hotkey="5Copy",
            commitment=ModelCommitment("copycat/guard", REV_A, "chute-2"),
            commit_block=900,
        )

        losers = _duplicate_losers([mirror, original])
        assert losers[0].hotkey == "5Orig"

    def test_genuinely_distinct_models_are_not_grouped(self) -> None:
        """Revision-only keying must not create false positives."""
        from prometheon.registry.validation import _duplicate_losers

        assert not _duplicate_losers(
            [
                MinerEntry(
                    uid=1,
                    hotkey="5A",
                    commitment=ModelCommitment("a/x", REV_A, "c1"),
                    commit_block=1,
                ),
                MinerEntry(
                    uid=2,
                    hotkey="5B",
                    commitment=ModelCommitment("b/y", REV_B, "c2"),
                    commit_block=2,
                ),
            ]
        )
