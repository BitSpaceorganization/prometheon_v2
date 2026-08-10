"""The eligibility pipeline: which miners may be scored on their model.

Seven checks run per miner, in a fixed order, and the first failure is the
reason reported. The order goes from cheapest and most local to most expensive
and most remote, so a miner with no commitment costs no requests, and the reason
a miner sees names the first thing they can fix.

    1. a commitment exists and decodes
    2. its revision is a 40-character commit SHA
    3. the Hugging Face repo passes the manifest and carries the canonical engine
    4. the deployed chute source normalises to an accepted wrapper hash
    5. the repo and revision *declared in that source* equal the commitment
    6. the chute is hot
    7. no earlier miner committed the same revision SHA

Check 5 is the one to understand. Reading the pinned repo and revision out of
the deployed source, rather than out of chute metadata, is what stops a miner
committing an immutable SHA on chain while serving a branch that they can move
after verification passes. Metadata does not reliably carry the revision at all;
the source does, and the wrapper hash in check 4 is what makes the source
trustworthy to read.

Check 7 resolves duplicates across *every* well-formed commitment, including the
ones that failed an earlier check. A copy must not become eligible because the
miner it copied happened to be cold that day. Otherwise the cheapest strategy in
the subnet is to publish someone else's model and wait for theirs to blink. It
groups on the **revision SHA alone**, never on the repo id: a Hugging Face repo
is a git repo, so mirroring one into another namespace preserves its commit SHAs
exactly, and a key that included the repo string put the original and its mirror
in separate groups and flagged neither.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from prometheon.canonical.hashes import (
    ACCEPTED_WRAPPER_HASHES,
    IMAGE_USERNAME,
    accepted_image_ids,
)
from prometheon.canonical.integrity import (
    extract_declared_variables,
    is_valid_revision,
    require_valid_revision,
    wrapper_hash,
)
from prometheon.chain.commitment import ModelCommitment
from prometheon.errors import (
    ChuteNotRunningError,
    CommitmentDecodeError,
    CommitmentError,
    CommitmentMismatchError,
    DuplicateModelError,
    ImageMismatchError,
    ManifestViolationError,
    RegistryError,
    RevisionFormatError,
    WrapperHashMismatchError,
)
from prometheon.registry.chutes import (
    MODERATE_PATH,
    ChuteInfo,
    ChutesClient,
    require_valid_chute_id,
)
from prometheon.registry.huggingface import HuggingFaceClient, require_valid_repo_id

_DECLARED_REPO: Final[str] = "PROMETHEON_HF_REPO"
_DECLARED_REVISION: Final[str] = "PROMETHEON_HF_REVISION"


class InvalidReason(str, Enum):
    """Why a miner is not eligible for the model pool.

    Values are the error codes from :mod:`prometheon.errors` rather than a
    parallel vocabulary, so a reason in an evaluation record, a log line, and a
    raised exception are the same string.
    """

    COMMITMENT_MISSING = CommitmentError.code
    COMMITMENT_MALFORMED = CommitmentDecodeError.code
    REVISION_NOT_SHA = RevisionFormatError.code
    MANIFEST_VIOLATION = ManifestViolationError.code
    IMAGE_MISMATCH = ImageMismatchError.code
    WRAPPER_HASH_MISMATCH = WrapperHashMismatchError.code
    COMMITMENT_MISMATCH = CommitmentMismatchError.code
    CHUTE_NOT_RUNNING = ChuteNotRunningError.code
    DUPLICATE_MODEL = DuplicateModelError.code
    #: Hugging Face or Chutes could not be reached. Not the miner's fault, but
    #: an unverifiable model is still not a scorable one.
    UPSTREAM_UNAVAILABLE = RegistryError.code


#: The block of a commitment nobody could read.
#:
#: It sorts **last**, and the direction is the whole point. Duplicate claims are
#: settled by ascending block, so a default of ``0`` meant "unknown" was
#: indistinguishable from "committed in the genesis block" and therefore *won*
#: every contest it entered. An unknown block cannot prove priority, so it must
#: lose to any block that can — and a caller that can make the read fail must
#: not be rewarded for it.
UNKNOWN_COMMIT_BLOCK: Final[int] = 1 << 62


@dataclass(frozen=True)
class MinerEntry:
    """One eligible miner as the chain presents it.

    ``commitment is None`` with no ``decode_error`` means the hotkey never
    committed. ``decode_error`` set means bytes were on chain but were not a
    Prometheon V2 commitment. Different failure, different fix, so the two are
    not collapsed.

    ``commit_block`` is the block the commitment was written in, read from the
    chain. It settles duplicates and nothing else.

    **It has no benign default.** It used to default to ``0``, and nothing in
    the production path ever set it — only the tests did — so every real
    duplicate contest was decided by ascending uid. That is the outcome
    :func:`_duplicate_losers` documents as the *strictly worse* attack: sit on a
    low uid and mirror anything good. The default is now
    :data:`UNKNOWN_COMMIT_BLOCK`, which sorts last, so the failure mode of
    forgetting to populate it is "this miner loses every tie" rather than "this
    miner wins every tie".
    """

    uid: int
    hotkey: str
    commitment: ModelCommitment | None = None
    decode_error: str | None = None
    commit_block: int = UNKNOWN_COMMIT_BLOCK


@dataclass(frozen=True)
class MinerValidation:
    """The verdict for one miner, with everything a later stage needs."""

    uid: int
    hotkey: str
    valid: bool
    reason: InvalidReason | None
    detail: str
    hf_repo: str
    hf_revision: str
    chute_id: str
    commit_block: int
    chute_slug: str
    endpoint_url: str
    wrapper_digest: str

    @property
    def moderate_url(self) -> str:
        """Where evaluation sends batches. Empty unless the miner is eligible.

        The gate is ``valid``. Gating on whether an endpoint was discovered was
        wrong: check 7 runs *after* the reachability check, so a duplicate came
        back ``valid=False`` carrying a live, addressable URL, and this
        docstring's promise was false for exactly the verdict that most needs
        it. Scoring a copy is what check 7 exists to prevent.
        """
        if not self.valid or not self.endpoint_url:
            return ""
        return f"{self.endpoint_url}{MODERATE_PATH}"


def _outcome(
    entry: MinerEntry,
    commitment: ModelCommitment | None,
    *,
    reason: InvalidReason | None,
    detail: str,
    chute_slug: str = "",
    endpoint_url: str = "",
    wrapper_digest: str = "",
) -> MinerValidation:
    return MinerValidation(
        uid=entry.uid,
        hotkey=entry.hotkey,
        valid=reason is None,
        reason=reason,
        detail=detail,
        hf_repo=commitment.hf_repo if commitment is not None else "",
        hf_revision=commitment.hf_revision if commitment is not None else "",
        chute_id=commitment.chute_id if commitment is not None else "",
        commit_block=entry.commit_block,
        chute_slug=chute_slug,
        endpoint_url=endpoint_url,
        wrapper_digest=wrapper_digest,
    )


def _hash_hostile_source(source: str) -> str:
    """``wrapper_hash`` with every failure a hostile string can cause contained.

    The source being hashed is fetched from a miner's own deployment, so it is
    attacker-controlled and must be treated as such. ``normalise_source`` walks
    the parse tree with ``ast.dump``, which recurses. A *perfectly valid* deeply
    nested expression (``x = 1+1+1+...``, 40 KB, far under the source size cap)
    parses fine and then blows the C stack.

    ``RecursionError`` is a ``RuntimeError``, not a ``RegistryError``, so it
    escaped the caller's handler, escaped ``validate()``, and aborted
    eligibility for **every miner in the cycle**. One miner could stop the whole
    subnet's daily scoring at no cost to themselves; they are ineligible either
    way.

    ``ValueError`` matters too: on Python 3.10 and 3.11, which this package
    supports, ``ast.parse`` on source containing a NUL byte raises ``ValueError``
    rather than ``SyntaxError``, so it slipped past the same handler.

    Anything that cannot be hashed is not the canonical wrapper, so all of these
    map to the same verdict.
    """
    try:
        return wrapper_hash(source)
    except RegistryError:
        raise
    except RecursionError as exc:
        raise WrapperHashMismatchError(
            "deployed source could not be normalised: it nests too deeply to "
            f"parse safely ({exc}). The canonical wrapper does not"
        ) from exc
    except Exception as exc:
        raise WrapperHashMismatchError(
            f"deployed source could not be normalised: {type(exc).__name__}: {exc}"
        ) from exc


def _extract_hostile_source(source: str) -> Mapping[str, str]:
    """``extract_declared_variables`` with the same containment.

    Reached only for source that already hashed to an accepted wrapper, so a
    failure here is close to impossible. But it walks the same attacker-supplied
    tree, and "close to impossible" is not a reason to let it abort the cycle.
    """
    try:
        return extract_declared_variables(source)
    except RegistryError:
        raise
    except Exception as exc:
        raise WrapperHashMismatchError(
            f"deployed source could not be read for its variables: {type(exc).__name__}: {exc}"
        ) from exc


def _duplicate_losers(entries: Sequence[MinerEntry]) -> dict[int, MinerEntry]:
    """Position in ``entries`` -> the miner that committed the same model first.

    **Grouped on the revision SHA alone, not on ``(repo, revision)``.** Putting
    the repo id in the key defeats the check against the cheapest attack there
    is.

    A Hugging Face model repo is a git repo. ``git clone --mirror`` followed by
    a push into another namespace preserves every commit SHA byte for byte, so
    a copycat can commit ``(copycat/guard, <the original's SHA>, own_chute)``
    and pass checks 1-6 honestly: the file manifest matches, the engine hash
    matches, and the deployed wrapper truthfully declares the copycat's own
    repo. Only the repo string differs, so keying on it put the original and
    the mirror in different groups and neither was ever flagged. One model
    could then hold rank 1 and rank 2 and collect 4500 of the 5000bp model
    pool using a second hotkey.

    Two genuinely distinct models sharing a 40-hex git commit SHA is not
    something that happens by accident, so this has no realistic false positive.

    **Known hazard: priority is the block of the *current* commitment, and the
    chain resets that on every re-commit.** A miner who redeploys their chute
    and re-commits the same revision under a new ``chute_id`` gets a fresh,
    higher block, so a copycat who mirrored them in the meantime now sorts
    first and takes the claim.

    Block ordering is kept regardless, because the alternative is worse. It is
    the *adversarially* correct key: a copycat cannot commit a revision SHA
    before that SHA exists, so against an attacker starting from nothing the
    original always sorts first. Ordering by uid instead would be stable under
    re-commit but would hand every claim to whoever registered earliest, and
    that is a strictly better attack: sit on a low uid and mirror anything good.
    Fixing the honest case properly needs a first-seen ledger, and a ledger
    that validators derive independently is a ledger they can disagree about,
    which turns a rare miner-facing hazard into routine weight divergence.

    The practical consequence for miners is documented in ``docs/miner.md``:
    re-committing re-dates your claim, so do not re-commit an unchanged model
    without reason.
    """
    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for position, entry in enumerate(entries):
        commitment = entry.commitment
        if entry.decode_error is not None or commitment is None:
            continue
        if not is_valid_revision(commitment.hf_revision):
            continue
        grouped.setdefault(commitment.hf_revision, []).append(
            (entry.commit_block, entry.uid, position)
        )

    losers: dict[int, MinerEntry] = {}
    for claims in grouped.values():
        if len(claims) < 2:
            continue
        # Lower uid breaks a same-block tie so every validator agrees; two
        # commitments really can land in one block.
        ordered = sorted(claims)
        winner = entries[ordered[0][2]]
        for _block, _uid, position in ordered[1:]:
            losers[position] = winner
    return losers


class ModelRegistry:
    """Runs the eligibility pipeline against live Hugging Face and Chutes state."""

    def __init__(self, *, huggingface: HuggingFaceClient, chutes: ChutesClient) -> None:
        self._huggingface = huggingface
        self._chutes = chutes

    def validate(self, entries: Sequence[MinerEntry]) -> list[MinerValidation]:
        """Verdicts for every entry, in the order given."""
        losers = _duplicate_losers(entries)
        # Duplicates and re-commitments of an unchanged model would otherwise
        # re-fetch the same repository once per miner. Only deterministic
        # verdicts are cached; an upstream outage is retried per miner because
        # it may well have cleared.
        repo_verdicts: dict[tuple[str, str], RegistryError | None] = {}
        return [
            self._validate_one(entry, losers.get(position), repo_verdicts)
            for position, entry in enumerate(entries)
        ]

    def _validate_one(
        self,
        entry: MinerEntry,
        duplicate_of: MinerEntry | None,
        repo_verdicts: dict[tuple[str, str], RegistryError | None],
    ) -> MinerValidation:
        # 1 -- the commitment exists, decoded, and names usable identifiers.
        if entry.decode_error is not None:
            return _outcome(
                entry,
                None,
                reason=InvalidReason.COMMITMENT_MALFORMED,
                detail=f"on-chain commitment could not be decoded: {entry.decode_error}",
            )
        commitment = entry.commitment
        if commitment is None:
            return _outcome(
                entry,
                None,
                reason=InvalidReason.COMMITMENT_MISSING,
                detail="no model commitment on chain",
            )
        try:
            require_valid_repo_id(commitment.hf_repo)
            require_valid_chute_id(commitment.chute_id)
        except RegistryError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.COMMITMENT_MALFORMED,
                detail=str(exc),
            )

        # 2 -- a moving reference is not a commitment. Re-checked here even
        # though the decoder enforces it, because eligibility must not rest on
        # another module continuing to hold an invariant it never promised the
        # registry.
        try:
            require_valid_revision(commitment.hf_revision)
        except RevisionFormatError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.REVISION_NOT_SHA, detail=str(exc)
            )

        # 3 -- the published repository.
        try:
            self._verify_repository(commitment.hf_repo, commitment.hf_revision, repo_verdicts)
        except ManifestViolationError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.MANIFEST_VIOLATION, detail=str(exc)
            )
        except RegistryError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.UPSTREAM_UNAVAILABLE, detail=str(exc)
            )

        try:
            info = self._chutes.fetch_chute(commitment.chute_id)
            source = self._chutes.deployed_source(info)
        except RegistryError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.UPSTREAM_UNAVAILABLE, detail=str(exc)
            )

        # 4 -- the deployed source is the canonical wrapper.
        try:
            digest = _hash_hostile_source(source)
        except RegistryError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.WRAPPER_HASH_MISMATCH,
                detail=f"deployed source is not parseable Python: {exc}",
                chute_slug=info.slug,
            )
        if digest not in ACCEPTED_WRAPPER_HASHES:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.WRAPPER_HASH_MISMATCH,
                detail=(
                    f"deployed source of chute {info.chute_id} normalises to {digest}, "
                    "which is not an accepted wrapper hash"
                ),
                chute_slug=info.slug,
                wrapper_digest=digest,
            )

        # 5 -- what is deployed is what was committed, read from the source.
        try:
            declared = _extract_hostile_source(source)
        except RegistryError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.WRAPPER_HASH_MISMATCH,
                detail=f"deployed source could not be read for its variables: {exc}",
                chute_slug=info.slug,
                wrapper_digest=digest,
            )
        mismatch = self._commitment_mismatch(commitment, declared)
        if mismatch is not None:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.COMMITMENT_MISMATCH,
                detail=mismatch,
                chute_slug=info.slug,
                wrapper_digest=digest,
            )

        # 6 -- the deployment runs the image the subnet published.
        #
        # The one thing about a runtime a validator can check. Everything above
        # proves which *bytes* are deployed; this is what makes them run
        # somewhere the miner did not assemble. The API exposes an image's
        # identity and never its contents, so an id from an account other than
        # the subnet's is not the subnet's image whatever it is called.
        accepted = accepted_image_ids()
        if not accepted:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.IMAGE_MISMATCH,
                detail=(
                    "no subnet image is configured, so no deployment can be verified to "
                    "run one. Refusing every model rather than accepting an image nobody "
                    "chose: the Chutes API exposes no image contents, so an unverified "
                    "image is an unknown runtime"
                ),
                chute_slug=info.slug,
                wrapper_digest=digest,
            )
        if info.image_id not in accepted or info.image_username != IMAGE_USERNAME:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.IMAGE_MISMATCH,
                detail=(
                    f"chute {info.chute_id} runs image {info.image_id or '(none reported)'} "
                    f"owned by {info.image_username or '(unknown)'}; this subnet scores only "
                    f"deployments running an image published by {IMAGE_USERNAME}"
                ),
                chute_slug=info.slug,
                wrapper_digest=digest,
            )

        # 7 -- the deployment can actually answer.
        try:
            endpoint = self._reachable_endpoint(info)
        except ChuteNotRunningError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.CHUTE_NOT_RUNNING,
                detail=str(exc),
                chute_slug=info.slug,
                wrapper_digest=digest,
            )

        # 8 -- an earlier commitment of the same model wins.
        if duplicate_of is not None:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.DUPLICATE_MODEL,
                detail=(
                    f"{commitment.hf_repo}@{commitment.hf_revision} was committed first by "
                    f"uid {duplicate_of.uid} ({duplicate_of.hotkey})"
                ),
                chute_slug=info.slug,
                endpoint_url=endpoint,
                wrapper_digest=digest,
            )

        return _outcome(
            entry,
            commitment,
            reason=None,
            detail="verified",
            chute_slug=info.slug,
            endpoint_url=endpoint,
            wrapper_digest=digest,
        )

    # -- steps ------------------------------------------------------------

    def _verify_repository(
        self,
        repo: str,
        revision: str,
        repo_verdicts: dict[tuple[str, str], RegistryError | None],
    ) -> None:
        key = (repo, revision)
        if key in repo_verdicts:
            cached = repo_verdicts[key]
            if cached is not None:
                raise cached
            return
        try:
            self._huggingface.verify_repository(repo, revision)
        except ManifestViolationError as exc:
            repo_verdicts[key] = exc
            raise
        repo_verdicts[key] = None

    def _commitment_mismatch(
        self, commitment: ModelCommitment, declared: Mapping[str, str]
    ) -> str | None:
        """The first disagreement between the deployed source and the chain."""
        deployed_repo = declared.get(_DECLARED_REPO, "")
        deployed_revision = declared.get(_DECLARED_REVISION, "")
        if deployed_repo != commitment.hf_repo:
            return (
                f"deployed wrapper serves repo {deployed_repo!r}, "
                f"the chain commits {commitment.hf_repo!r}"
            )
        if deployed_revision != commitment.hf_revision:
            return (
                f"deployed wrapper serves revision {deployed_revision!r}, "
                f"the chain commits {commitment.hf_revision!r}"
            )
        return None

    def _reachable_endpoint(self, info: ChuteInfo) -> str:
        if not info.hot:
            raise ChuteNotRunningError(
                f"chute {info.chute_id} ({info.slug}) is not hot, so it cannot serve /moderate"
            )
        try:
            return self._chutes.endpoint_url(info.slug)
        except RegistryError as exc:
            # A chute we cannot address is a chute that cannot be evaluated,
            # which is the same outcome for the miner as one that is not hot.
            raise ChuteNotRunningError(str(exc)) from exc


def eligible_miners(results: Iterable[MinerValidation]) -> list[MinerValidation]:
    """Only the miners that may be scored on their model."""
    return [result for result in results if result.valid]


__all__ = [
    "InvalidReason",
    "MinerEntry",
    "MinerValidation",
    "ModelRegistry",
    "eligible_miners",
]
