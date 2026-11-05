"""The eligibility pipeline: which frozen models a validator will run.

The checks run per model, in a fixed order, and the first failure is the reason
reported. The order goes from cheapest and most local to most remote, so a
model with no commitment costs no requests, and the reason names the first
thing its miner can fix.

    1. the commitment is well formed
    2. its revision is a 40-character commit SHA
    3. the repository passes the manifest: permitted files only, weights in
       Git LFS, a config, at most a few hundred weight files
    4. the architecture is one the evaluation runtime's transformers can load
    5. the weights, and everything else, fit the size ceilings
    6. the weights fingerprint matches the one the epoch froze
    7. no earlier commitment holds the same weights

Checks 3 to 6 come from one Hugging Face listing and one small config read,
before anything is downloaded.

**Duplicates are decided by file content (7).** v2 grouped on the git revision
SHA, so one extra commit on a copied model gave it a new SHA and it passed (B7).
The fingerprint is the sha256 over the sorted Git LFS sha256s of the weight
files, read from the API at the frozen revision, so renaming a shard or adding a
commit changes nothing. Only models that passed checks 1 to 6 compete, so a
broken repository cannot block a working copy of the same weights. The earliest
``commit_block`` wins, then the lower hotkey; the hub publishes the first block
at which each hotkey committed each fingerprint, so re-committing an unchanged
model does not lose the claim.

**Who is unavailable matters.** A repository that is gone or private is the
miner's loss on every validator alike. Hugging Face failing to answer one
validator is that validator's fault, reported separately so the cycle can stop
rather than score a miner out of the day.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Final

from prometheon.chain.commitment import ModelCommitment
from prometheon.errors import (
    CommitmentDecodeError,
    CommitmentError,
    DuplicateModelError,
    FingerprintMismatchError,
    ManifestViolationError,
    ModelTooLargeError,
    RegistryError,
    RegistryUnavailableError,
    RepoUnavailableError,
    RevisionFormatError,
)
from prometheon.registry.huggingface import (
    HuggingFaceClient,
    RepoSnapshot,
    require_valid_repo_id,
    weights_fingerprint,
)
from prometheon.registry.loadability import ArchitectureUnsupportedError, require_loadable
from prometheon.revision import require_valid_revision


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
    #: The architecture is one the evaluation runtime cannot load at all.
    ARCHITECTURE_UNSUPPORTED = ArchitectureUnsupportedError.code
    #: The weights exceed what every validator is expected to be able to run.
    MODEL_TOO_LARGE = ModelTooLargeError.code
    DUPLICATE_MODEL = DuplicateModelError.code
    #: The weights at the frozen revision are not the ones the epoch names.
    FINGERPRINT_MISMATCH = FingerprintMismatchError.code
    #: The repository or revision is gone, private or gated.
    REPO_UNAVAILABLE = RepoUnavailableError.code
    #: Hugging Face could not be reached. Never the miner's fault: a cycle that
    #: sees this stops rather than scoring the model out.
    UPSTREAM_UNAVAILABLE = RegistryUnavailableError.code


#: The block of a commitment nobody could read.
#:
#: It sorts **last**, and the direction is the whole point. Duplicate claims are
#: settled by ascending block, so a default of ``0`` meant "unknown" was
#: indistinguishable from "committed in the genesis block" and therefore *won*
#: every contest it entered. An unknown block cannot prove priority, so it must
#: lose to any block that can, and a caller that can make the read fail must
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
    the production path ever set it, only the tests did, so every real
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
    #: The fingerprint the epoch froze. Empty when there is none to compare
    #: against, as in ``prometheon model verify`` before a model is committed.
    expected_fingerprint: str = ""


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
    commit_block: int
    #: What the checkpoint declares itself to be, once check 4 has read it.
    model_type: str = ""
    #: Total weight bytes, as the API reported them for check 5.
    weight_bytes: int = 0
    #: Everything that is not weights, in bytes.
    non_weight_bytes: int = 0
    #: The weights fingerprint, once check 6 has computed it.
    fingerprint: str = ""


def _outcome(
    entry: MinerEntry,
    commitment: ModelCommitment | None,
    *,
    reason: InvalidReason | None,
    detail: str,
    model_type: str = "",
    snapshot: RepoSnapshot | None = None,
    fingerprint: str = "",
) -> MinerValidation:
    return MinerValidation(
        uid=entry.uid,
        hotkey=entry.hotkey,
        valid=reason is None,
        reason=reason,
        detail=detail,
        hf_repo=commitment.hf_repo if commitment is not None else "",
        hf_revision=commitment.hf_revision if commitment is not None else "",
        commit_block=entry.commit_block,
        model_type=model_type,
        weight_bytes=snapshot.weight_bytes if snapshot is not None else 0,
        non_weight_bytes=snapshot.non_weight_bytes if snapshot is not None else 0,
        fingerprint=fingerprint,
    )


def _registry_reason(exc: RegistryError) -> InvalidReason:
    if isinstance(exc, RepoUnavailableError):
        return InvalidReason.REPO_UNAVAILABLE
    if isinstance(exc, ManifestViolationError):
        return InvalidReason.MANIFEST_VIOLATION
    return InvalidReason.UPSTREAM_UNAVAILABLE


def _resolve_duplicates(results: list[MinerValidation]) -> list[MinerValidation]:
    """Among models that passed every other check, the earliest holds its weights.

    Ordered by ``(commit_block, hotkey)``: two commitments really can land in one
    block, and the hotkey settles that the same way on every validator.
    """
    holders: dict[str, MinerValidation] = {}
    for result in sorted(
        (result for result in results if result.valid),
        key=lambda result: (result.commit_block, result.hotkey),
    ):
        holders.setdefault(result.fingerprint, result)

    resolved: list[MinerValidation] = []
    for result in results:
        holder = holders.get(result.fingerprint) if result.valid else None
        if holder is None or holder.hotkey == result.hotkey:
            resolved.append(result)
            continue
        resolved.append(
            replace(
                result,
                valid=False,
                reason=InvalidReason.DUPLICATE_MODEL,
                detail=(
                    f"{result.hf_repo}@{result.hf_revision} carries the same weights as "
                    f"{holder.hf_repo}@{holder.hf_revision}, committed first by {holder.hotkey}"
                ),
            )
        )
    return resolved


class ModelRegistry:
    """Runs the eligibility pipeline against live Hugging Face state.

    The ceilings come from ``[evaluation]``. They describe the hardware and the
    disk every validator agreed to provide, which is a subnet policy decision
    and lives in config with the others that move emissions.
    """

    def __init__(
        self,
        *,
        huggingface: HuggingFaceClient,
        max_weight_bytes: int = 0,
        max_non_weight_bytes: int = 0,
    ) -> None:
        self._huggingface = huggingface
        self._max_weight_bytes = max_weight_bytes
        self._max_non_weight_bytes = max_non_weight_bytes

    def validate(self, entries: Sequence[MinerEntry]) -> list[MinerValidation]:
        """Verdicts for every entry, in the order given."""
        # A listing is deterministic for a commit SHA, so two entries on one
        # revision share it. An outage is not cached: it may have cleared.
        listings: dict[tuple[str, str], RepoSnapshot] = {}
        return _resolve_duplicates([self._validate_one(entry, listings) for entry in entries])

    def _validate_one(
        self, entry: MinerEntry, listings: dict[tuple[str, str], RepoSnapshot]
    ) -> MinerValidation:
        # 1 -- the commitment exists, decoded, and names usable identifiers.
        if entry.decode_error is not None:
            return _outcome(
                entry,
                None,
                reason=InvalidReason.COMMITMENT_MALFORMED,
                detail=f"commitment could not be decoded: {entry.decode_error}",
            )
        commitment = entry.commitment
        if commitment is None:
            return _outcome(
                entry,
                None,
                reason=InvalidReason.COMMITMENT_MISSING,
                detail="no model commitment",
            )
        try:
            require_valid_repo_id(commitment.hf_repo)
        except RegistryError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.COMMITMENT_MALFORMED, detail=str(exc)
            )

        # 2 -- a moving reference is not a commitment.
        try:
            require_valid_revision(commitment.hf_revision)
        except RevisionFormatError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.REVISION_NOT_SHA, detail=str(exc)
            )

        # 3 -- the repository at that commit.
        key = (commitment.hf_repo, commitment.hf_revision)
        try:
            snapshot = listings.get(key)
            if snapshot is None:
                snapshot = self._huggingface.verify_repository(*key)
                listings[key] = snapshot
        except RegistryError as exc:
            return _outcome(entry, commitment, reason=_registry_reason(exc), detail=str(exc))

        # 4 -- the evaluation runtime can load this architecture at all.
        try:
            model_type = require_loadable(
                self._huggingface, commitment.hf_repo, commitment.hf_revision
            )
        except ArchitectureUnsupportedError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.ARCHITECTURE_UNSUPPORTED,
                detail=str(exc),
                snapshot=snapshot,
            )
        except RegistryError as exc:
            return _outcome(
                entry, commitment, reason=_registry_reason(exc), detail=str(exc), snapshot=snapshot
            )
        model_type = model_type or ""

        # 5 -- the weights fit the hardware every validator is expected to have,
        # and nothing else in the repository is big enough to fill their disks.
        too_large = self._size_problem(commitment, snapshot)
        if too_large:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.MODEL_TOO_LARGE,
                detail=too_large,
                model_type=model_type,
                snapshot=snapshot,
            )

        # 6 -- the weights are the ones the day froze.
        try:
            fingerprint = weights_fingerprint(snapshot)
        except ManifestViolationError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.MANIFEST_VIOLATION,
                detail=str(exc),
                model_type=model_type,
                snapshot=snapshot,
            )
        if entry.expected_fingerprint and fingerprint != entry.expected_fingerprint:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.FINGERPRINT_MISMATCH,
                detail=(
                    f"{commitment.hf_repo}@{commitment.hf_revision} fingerprints as "
                    f"{fingerprint[:12]}, but the epoch froze {entry.expected_fingerprint[:12]}"
                ),
                model_type=model_type,
                snapshot=snapshot,
                fingerprint=fingerprint,
            )

        return _outcome(
            entry,
            commitment,
            reason=None,
            detail="verified",
            model_type=model_type,
            snapshot=snapshot,
            fingerprint=fingerprint,
        )

    def _size_problem(self, commitment: ModelCommitment, snapshot: RepoSnapshot) -> str:
        where = f"{commitment.hf_repo}@{commitment.hf_revision}"
        if self._max_weight_bytes and snapshot.weight_bytes > self._max_weight_bytes:
            return (
                f"{where} carries {snapshot.weight_bytes / 1024**3:.1f} GiB of weights, over the "
                f"{self._max_weight_bytes / 1024**3:.1f} GiB ceiling every validator must be "
                "able to load"
            )
        if self._max_non_weight_bytes and snapshot.non_weight_bytes > self._max_non_weight_bytes:
            return (
                f"{where} carries {snapshot.non_weight_bytes / 1024**2:.1f} MiB of files that are "
                f"not weights, over the {self._max_non_weight_bytes / 1024**2:.1f} MiB ceiling. "
                "Every validator downloads all of it"
            )
        return ""


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
