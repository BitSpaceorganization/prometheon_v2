"""The eligibility pipeline: which miners may be scored on their model.

Six checks run per miner, in a fixed order, and the first failure is the
reason reported. The order goes from cheapest and most local to most expensive
and most remote, so a miner with no commitment costs no requests, and the reason
a miner sees names the first thing they can fix.

    1. a commitment exists and decodes
    2. its revision is a 40-character commit SHA
    3. the Hugging Face repo passes the manifest: weights, config and tokenizer
    4. the architecture is one the evaluation image's transformers can load
    5. the weights fit the size ceiling every validator has to run them on
    6. no earlier miner committed the same revision SHA

An attack this pipeline no longer has to defend against is worth naming.
While miners served their own models, a commitment could pin an immutable SHA
on chain while the deployment served a branch the miner could move after
verification passed, so the *deployed source* had to be read and hashed to
catch it. Validators now fetch the checkpoint at the committed SHA themselves.
There is no deployment to lie about: what is scored is what was committed,
because the validator is the one that resolved it.

Checks 4 and 5 are the ones to understand now. Both exist because every
validator must run every eligible model on its own hardware, so a model that
cannot load, or cannot fit, is not a private mistake by one miner -- it is work
the whole subnet attempts and fails at simultaneously. Both are answered from
the Hugging Face API before anything is downloaded.

Check 6 resolves duplicates across *every* well-formed commitment, including the
ones that failed an earlier check. A copy must not become eligible because the
miner it copied happened to be cold that day. Otherwise the cheapest strategy in
the subnet is to publish someone else's model and wait for theirs to blink. It
groups on the **revision SHA alone**, never on the repo id: a Hugging Face repo
is a git repo, so mirroring one into another namespace preserves its commit SHAs
exactly, and a key that included the repo string put the original and its mirror
in separate groups and flagged neither.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Final

from prometheon.chain.commitment import ModelCommitment
from prometheon.errors import (
    CommitmentDecodeError,
    CommitmentError,
    DuplicateModelError,
    ManifestViolationError,
    ModelTooLargeError,
    RegistryError,
    RevisionFormatError,
)
from prometheon.registry.huggingface import HuggingFaceClient, require_valid_repo_id
from prometheon.registry.loadability import ArchitectureUnsupportedError, require_loadable
from prometheon.revision import is_valid_revision, require_valid_revision


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
    #: Hugging Face could not be reached. Not the miner's fault, but an
    #: unverifiable model is still not a scorable one.
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
    commit_block: int
    #: What the checkpoint declares itself to be, once check 4 has read it.
    model_type: str = ""
    #: Total weight bytes, as the API reported them for check 5.
    weight_bytes: int = 0


def _outcome(
    entry: MinerEntry,
    commitment: ModelCommitment | None,
    *,
    reason: InvalidReason | None,
    detail: str,
    model_type: str = "",
    weight_bytes: int = 0,
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
        weight_bytes=weight_bytes,
    )


def _duplicate_losers(entries: Sequence[MinerEntry]) -> dict[int, MinerEntry]:
    """Position in ``entries`` -> the miner that committed the same model first.

    **Grouped on the revision SHA alone, not on ``(repo, revision)``.** Putting
    the repo id in the key defeats the check against the cheapest attack there
    is.

    A Hugging Face model repo is a git repo. ``git clone --mirror`` followed by
    a push into another namespace preserves every commit SHA byte for byte, so
    a copycat can commit ``(copycat/guard, <the original's SHA>)``
    and pass checks 1-5 honestly: it is the same tree, so the file manifest
    matches, the architecture is loadable, and the weights are the same size.
    Only the repo string differs, so keying on it put the original and
    the mirror in different groups and neither was ever flagged. One model
    could then hold rank 1 and rank 2 and collect 4500 of the 5000bp model
    pool using a second hotkey.

    Two genuinely distinct models sharing a 40-hex git commit SHA is not
    something that happens by accident, so this has no realistic false positive.

    **Known hazard: priority is the block of the *current* commitment, and the
    chain resets that on every re-commit.** A miner who re-commits the same
    revision after republishing gets a fresh,
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
    """Runs the eligibility pipeline against live Hugging Face state.

    ``max_weight_bytes`` is the ceiling from ``[evaluation]``. It is a
    constructor argument rather than a module constant because it describes the
    hardware validators agreed to run, which is a subnet policy decision and
    lives in config with the others that move emissions.
    """

    def __init__(self, *, huggingface: HuggingFaceClient, max_weight_bytes: int = 0) -> None:
        self._huggingface = huggingface
        self._max_weight_bytes = max_weight_bytes

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

        # 4 -- the evaluation runtime can load this architecture at all.
        #
        # Answered from `config.json`, before anything is downloaded. An
        # architecture the pinned transformers does not know is not a slow
        # model or a bad model: it is one that raises at load, on every
        # validator, every day, until the miner commits something else.
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
            )
        except RegistryError as exc:
            return _outcome(
                entry, commitment, reason=InvalidReason.UPSTREAM_UNAVAILABLE, detail=str(exc)
            )

        # 5 -- the weights fit the hardware every validator is expected to have.
        #
        # Also answered from the API rather than by downloading. A model over
        # the ceiling is not one validator's problem: every validator on the
        # subnet would pull it and fail to fit it, at the same time, having
        # already paid for the bandwidth.
        try:
            weight_bytes = self._huggingface.weight_bytes(
                commitment.hf_repo, commitment.hf_revision
            )
        except RegistryError as exc:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.UPSTREAM_UNAVAILABLE,
                detail=str(exc),
                model_type=model_type or "",
            )
        if self._max_weight_bytes and weight_bytes > self._max_weight_bytes:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.MODEL_TOO_LARGE,
                detail=(
                    f"{commitment.hf_repo}@{commitment.hf_revision} carries "
                    f"{weight_bytes / 1024**3:.1f} GiB of weights, over the "
                    f"{self._max_weight_bytes / 1024**3:.1f} GiB ceiling every validator "
                    "must be able to load"
                ),
                model_type=model_type or "",
                weight_bytes=weight_bytes,
            )

        # 6 -- an earlier commitment of the same model wins.
        if duplicate_of is not None:
            return _outcome(
                entry,
                commitment,
                reason=InvalidReason.DUPLICATE_MODEL,
                detail=(
                    f"{commitment.hf_repo}@{commitment.hf_revision} was committed first by "
                    f"uid {duplicate_of.uid} ({duplicate_of.hotkey})"
                ),
                model_type=model_type or "",
                weight_bytes=weight_bytes,
            )

        return _outcome(
            entry,
            commitment,
            reason=None,
            detail="verified",
            model_type=model_type or "",
            weight_bytes=weight_bytes,
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
