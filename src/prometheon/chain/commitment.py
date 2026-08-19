"""The on-chain model commitment: a compact, versioned, 128-byte-safe encoding.

A commitment answers one question. *Which model is this hotkey being scored on
today?* It answers with two values: the Hugging Face repo and the 40-hex
revision SHA. Freezing them on chain is what makes the day's evaluation
reproducible by anyone: a third party reads the commitment, fetches that
revision, and recomputes.

**Format 2 carries no deployment id.** Format 1 named a Chutes chute, because
the miner served the model and validators called it. Validators now download
the weights and run them, so there is nothing to address and nothing to prove
about a deployment -- the repo and the revision are the whole submission. A
format 1 payload is refused rather than partially read: it describes a
deployment that is no longer evaluated, and silently scoring its repo would
score a model the miner never resubmitted under the current rules.

The 128-byte budget drives the rest of the design. Substrate stores a
commitment as a ``Raw`` blob and the largest variant is ``Raw128``, so anything
longer is rejected by the chain, not by us. Size costs twice over, because the
commitments pallet also charges every write against a per-hotkey byte quota
that resets each epoch, so a bloated payload buys fewer corrections on the day
a miner needs one.

The revision is 160 bits of entropy wearing 40 characters of hex. Base64url of
the raw 20 bytes is 27 characters, so 13 are saved.

Layout, all ASCII::

    P 2 <ver> <revision:27> <hf_repo…>
    │ │   │         │           └─ variable, runs to the end
    │ │   │         └───────────── base64url of 20 raw bytes
    │ │   └─────────────────────── format version, '2'
    └─┴─────────────────────────── magic

Size budget::

    magic + version                       3
    revision                             27
    ──────────────────────────────────── ──
    fixed cost                           30   leaving 98 bytes for hf_repo

``hf_repo`` runs last so it needs no length prefix or terminator: everything
ahead of it is fixed-width.

Nothing here is signed. The extrinsic that writes a commitment is already
signed by the hotkey, so authorship is chain-attested; a 64-byte SR25519
signature would also consume half the budget to prove something the chain
proves for free.

The encoding is canonical: one commitment has exactly one valid encoding, so
two validators reading the same commitment cannot disagree about what it says.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Any, Final

from prometheon.chain.subtensor import parse_extrinsic_response
from prometheon.errors import CommitmentDecodeError, CommitmentError
from prometheon.revision import is_valid_revision

#: The largest ``Raw`` variant substrate will store for a commitment.
COMMITMENT_MAX_BYTES: Final[int] = 128

_MAGIC: Final[str] = "P2"
_FORMAT_VERSION: Final[str] = "2"

#: Format 1 named a Chutes deployment. Kept only so its payloads can be
#: refused by name rather than as "unreadable".
_CHUTES_ERA_VERSION: Final[str] = "1"

_HEADER_CHARS: Final[int] = 3
_REVISION_CHARS: Final[int] = 27
_REVISION_BYTES: Final[int] = 20

#: Hugging Face allows ``namespace/name`` or a bare canonical name, each
#: starting alphanumeric. Anything else is not a repo we could fetch anyway.
_HF_REPO_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)


@dataclass(frozen=True)
class ModelCommitment:
    """The two values a miner freezes on chain for a cycle.

    Validated on construction, so an instance in hand is always publishable.
    Every rejection happens at the boundary rather than at submission time,
    when the miner has already lost the day.
    """

    hf_repo: str
    hf_revision: str

    def __post_init__(self) -> None:
        if not _HF_REPO_RE.match(self.hf_repo or ""):
            raise CommitmentError(f"hf_repo must look like 'namespace/name', got {self.hf_repo!r}")
        if not is_valid_revision(self.hf_revision):
            # Not RevisionFormatError: a commitment failure has to carry a
            # chain.commitment_* code so operators reach the commitment
            # runbook, not the registry one.
            raise CommitmentError(
                "hf_revision must be a 40-character lowercase hex SHA "
                f"(a branch or tag is not a commitment), got {self.hf_revision!r}"
            )


def encode_commitment(commitment: ModelCommitment) -> str:
    """Encode ``commitment`` to the ASCII payload written on chain.

    Raises :class:`CommitmentError` if the result would exceed the chain's byte
    budget. Given the fixed costs above, that only happens for an unusually
    long repo id.
    """
    revision_field = _b64url_encode(bytes.fromhex(commitment.hf_revision))
    text = f"{_MAGIC}{_FORMAT_VERSION}{revision_field}{commitment.hf_repo}"
    size = len(text.encode("ascii"))
    if size > COMMITMENT_MAX_BYTES:
        raise CommitmentError(
            f"commitment encodes to {size} bytes, over the chain's "
            f"{COMMITMENT_MAX_BYTES}-byte limit; shorten hf_repo "
            f"({len(commitment.hf_repo)} chars) by at least "
            f"{size - COMMITMENT_MAX_BYTES}"
        )
    return text


def decode_commitment(payload: str | bytes) -> ModelCommitment:
    """Decode an on-chain payload, or raise :class:`CommitmentDecodeError`.

    Every rejection path is explicit. A commitment that does not decode means
    the hotkey is not evaluated this cycle, so the failure has to name what was
    wrong with it. A bare "malformed" leaves a miner nothing to fix.
    """
    text = _as_ascii(payload)

    if not text:
        raise CommitmentDecodeError("commitment is empty")
    if len(text) > COMMITMENT_MAX_BYTES:
        raise CommitmentDecodeError(
            f"commitment is {len(text)} bytes, over the {COMMITMENT_MAX_BYTES}-byte limit"
        )
    if not text.startswith(_MAGIC):
        raise CommitmentDecodeError(
            f"not a Prometheon V2 model commitment: expected magic {_MAGIC!r}"
        )
    if len(text) < _HEADER_CHARS + _REVISION_CHARS:
        raise CommitmentDecodeError(
            f"commitment is truncated: {len(text)} bytes cannot hold a header and a revision"
        )

    version = text[2]
    if version == _CHUTES_ERA_VERSION:
        raise CommitmentDecodeError(
            "this is a format 1 commitment, which named a Chutes deployment. Models are "
            "now downloaded and run by validators, so there is no deployment to score: "
            "re-commit with `prometheon model commit` to publish a format 2 payload"
        )
    if version != _FORMAT_VERSION:
        raise CommitmentDecodeError(
            f"commitment format version {version!r} is not readable by this build "
            f"(which reads {_FORMAT_VERSION!r}); upgrade the validator"
        )

    revision = _b64url_decode(
        text[_HEADER_CHARS : _HEADER_CHARS + _REVISION_CHARS],
        expected_bytes=_REVISION_BYTES,
        field="hf_revision",
    ).hex()
    hf_repo = text[_HEADER_CHARS + _REVISION_CHARS :]

    if not hf_repo:
        raise CommitmentDecodeError("commitment carries no hf_repo")

    try:
        return ModelCommitment(hf_repo=hf_repo, hf_revision=revision)
    except CommitmentError as exc:
        raise CommitmentDecodeError(f"commitment decoded to invalid values: {exc}") from exc


def commitment_size_bytes(commitment: ModelCommitment) -> int:
    """What ``commitment`` will occupy on chain. For a pre-flight check."""
    return len(encode_commitment(commitment).encode("ascii"))


def publish_commitment(
    subtensor: Any, *, wallet: Any, netuid: int, commitment: ModelCommitment
) -> str:
    """Write ``commitment`` to the chain and return the payload that was written.

    Returning the payload rather than nothing lets the miner CLI print exactly
    what a validator will read back, which is the only thing that settles a
    "my commitment is wrong" report.
    """
    payload = encode_commitment(commitment)

    accessor = getattr(subtensor, "set_commitment", None)
    if accessor is None:
        raise CommitmentError(
            f"the installed bittensor SDK exposes no Subtensor.set_commitment, so "
            f"this hotkey cannot publish a model commitment on netuid={netuid}; "
            "this runtime drives the 10.x accessor API, pin 'bittensor>=10.5,<11'"
        )

    try:
        result = accessor(wallet=wallet, netuid=netuid, data=payload)
    except Exception as exc:
        raise CommitmentError(
            f"could not publish the model commitment on netuid={netuid}: {exc}"
        ) from exc

    # Shared with weight submission because both writes return the same
    # `ExtrinsicResponse` and both previously had their own copy of this
    # normalisation. One of those copies scored a `(False, "reason")` tuple as
    # success, since a tuple has no `.success` to read.
    try:
        parse_extrinsic_response(result, operation="set_commitment", error=CommitmentError)
    except CommitmentError as exc:
        raise CommitmentError(
            f"{exc} — note that each commitment consumes at least 100 bytes of the "
            "hotkey's per-epoch byte quota on the commitments pallet, so a "
            "rejected write often means the quota is spent; retry after the "
            "next epoch"
        ) from exc
    return payload


def read_commitment(
    subtensor: Any, *, netuid: int, hotkey: str, uid: int
) -> ModelCommitment | None:
    """Read one miner's commitment, or ``None`` if it has never committed.

    The SDK accessor is keyed by UID, and a UID is only a slot. UIDs are
    recycled the moment a neuron deregisters, so a UID resolved at the start of
    a cycle can belong to a different hotkey by the time it is read. That is
    how one miner's model gets attributed to whoever inherited its slot. So the
    caller passes ``uid`` resolved from the *same* metagraph snapshot the cycle
    is scoring, with ``hotkey`` alongside it so failures name the identity
    rather than the slot.

    "Never committed" is a normal state and returns ``None``. A payload that is
    present but unreadable raises.
    """
    try:
        raw = subtensor.get_commitment(netuid=netuid, uid=uid)
    except Exception as exc:
        raise CommitmentError(
            f"could not read the commitment for {hotkey!r} (uid={uid}) on netuid={netuid}: {exc}"
        ) from exc

    # Newer clients return a record wrapping the payload; older ones return the
    # payload itself.
    if raw is not None and not isinstance(raw, (str, bytes, bytearray)):
        raw = getattr(raw, "data", None)

    if raw is None:
        return None
    if isinstance(raw, str) and not raw.strip():
        return None
    if isinstance(raw, (bytes, bytearray)) and not bytes(raw):
        return None
    return decode_commitment(raw)


def read_commitment_block(subtensor: Any, *, netuid: int, hotkey: str) -> int:
    """The block a hotkey's commitment was written in.

    This is what settles a duplicate-model contest, and it has to come from the
    **chain**. The subnet DB layer mirrors a ``block`` on its own commitment
    records and it would have been one line to read it from there — but
    duplicate resolution is an anti-gaming mechanism, and sourcing it from the
    service being audited would make the anti-gaming property depend on the one
    component a validator is otherwise careful never to trust. The chain is the
    authority on commitments; this reads the authority.

    Keyed by ``hotkey_ss58`` rather than by uid, unlike
    :func:`read_commitment`. A uid is a recycled slot, so a metadata read keyed
    by uid can answer about whoever inherited it; a hotkey is identity.

    Returns :data:`~prometheon.registry.validation.UNKNOWN_COMMIT_BLOCK` when
    the chain has no readable block for this hotkey. That value sorts *last*, so
    a hotkey whose block cannot be established loses every duplicate contest it
    enters rather than winning them.
    """
    from prometheon.registry.validation import UNKNOWN_COMMIT_BLOCK

    accessor = getattr(subtensor, "get_commitment_metadata", None)
    if accessor is None:
        raise CommitmentError(
            "the installed bittensor SDK exposes no Subtensor.get_commitment_metadata, "
            "so the block a commitment was written in cannot be read. Without it, "
            "duplicate-model claims fall back to uid order, which hands every claim "
            "to whoever registered earliest; this runtime drives the 10.x accessor "
            "API, pin 'bittensor>=10.5,<11'"
        )

    try:
        metadata = accessor(netuid=netuid, hotkey_ss58=hotkey)
    except Exception as exc:
        raise CommitmentError(
            f"could not read commitment metadata for {hotkey!r} on netuid={netuid}: {exc}"
        ) from exc

    # The accessor is typed `str | CommitmentOfResponse`: the plain-string form
    # carries the payload and no block at all. Absent rather than zero, because
    # zero is a real block number and would win.
    block = getattr(metadata, "block", None)
    if block is None:
        return UNKNOWN_COMMIT_BLOCK
    try:
        return int(block)
    except (TypeError, ValueError):
        return UNKNOWN_COMMIT_BLOCK


def _as_ascii(payload: str | bytes) -> str:
    """Normalise the shapes a commitment arrives in to plain ASCII text.

    Some SDK paths hand back the decoded string, others the raw bytes, and
    others the ``0x``-prefixed hex substrate actually stores. All three are the
    same commitment.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            text = bytes(payload).decode("ascii")
        except UnicodeDecodeError as exc:
            raise CommitmentDecodeError(f"commitment bytes are not ASCII: {exc}") from exc
    else:
        text = payload

    if text.startswith("0x"):
        try:
            text = bytes.fromhex(text[2:]).decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise CommitmentDecodeError(
                f"commitment looked hex-encoded but did not decode to ASCII: {exc}"
            ) from exc

    if not text.isascii():
        raise CommitmentDecodeError("commitment contains non-ASCII characters")
    return text


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str, *, expected_bytes: int, field: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    try:
        raw = base64.b64decode(padded, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CommitmentDecodeError(f"{field} is not valid base64url: {exc}") from exc

    if len(raw) != expected_bytes:
        raise CommitmentDecodeError(
            f"{field} decoded to {len(raw)} bytes, expected {expected_bytes}"
        )
    if _b64url_encode(raw) != text:
        # The final base64 character of a 20- or 16-byte value carries padding
        # bits that must be zero. Accepting a non-zero variant would give one
        # commitment several spellings.
        raise CommitmentDecodeError(f"{field} is not canonically base64url encoded")
    return raw


__all__ = [
    "COMMITMENT_MAX_BYTES",
    "ModelCommitment",
    "commitment_size_bytes",
    "decode_commitment",
    "encode_commitment",
    "publish_commitment",
    "read_commitment",
    "read_commitment_block",
]
