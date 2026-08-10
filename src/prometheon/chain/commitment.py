"""The on-chain model commitment: a compact, versioned, 128-byte-safe encoding.

A commitment answers one question. *Which model is this hotkey being scored on
today?* It answers with three values: the Hugging Face repo, the 40-hex
revision SHA, and the Chutes deployment id. Freezing them on chain is what
makes the day's evaluation reproducible by anyone: a third party reads the
commitment, fetches that revision, and recomputes.

The 128-byte budget drives the rest of the design. Substrate stores a
commitment as a ``Raw`` blob and the largest variant is ``Raw128``, so anything
longer is rejected by the chain, not by us. Size costs twice over, because the
commitments pallet also charges every write against a per-hotkey byte quota
that resets each epoch, so a bloated payload buys fewer corrections on the day
a miner needs one. A naive JSON dump does not fit::

    {"chute_id":"b4e8a2f1-6c3d-4e5a-9b7f-1d2c3e4f5a6b","hf_repo":"prometheon-labs/
     moderation-guard-8b","hf_revision":"<40 hex>"}          = 156 bytes

156 bytes for 111 bytes of information, of which 45 are field names repeated in
every commitment on the subnet. Two substitutions recover the headroom:

- The revision is 160 bits of entropy wearing 40 characters of hex. Base64url
  of the raw 20 bytes is 27 characters, so 13 are saved.
- A Chutes deployment id is a UUID: 128 bits wearing 36 characters. Base64url
  of the raw 16 bytes is 22 characters, so 14 are saved.

Layout, all ASCII::

    P 2 <ver> <mode> <revision:27> <chute> <hf_repo…>
    │ │   │      │          │         │        └─ variable, runs to the end
    │ │   │      │          │         └────────── mode 'u': 22 chars, fixed
    │ │   │      │          │                     mode 't': literal + ':'
    │ │   │      │          └──────────────────── base64url of 20 raw bytes
    │ │   │      └─────────────────────────────── 'u' compact UUID | 't' literal
    │ │   └────────────────────────────────────── format version, '1'
    └─┴────────────────────────────────────────── magic

Size budget (mode ``u``)::

    magic + version + mode                4
    revision                             27
    chute id                             22
    ──────────────────────────────────── ──
    fixed cost                           53   leaving 75 bytes for hf_repo

The worked example above encodes to **88 bytes**. Mode ``t`` costs one
terminator plus the literal id, leaving 60 bytes of repo for a 36-character id.

``hf_repo`` runs last so it needs no length prefix or terminator. The fields
ahead of it are either fixed-width or terminated by a character no legal repo
id, chute id, or base64url digit contains.

Nothing here is signed. The extrinsic that writes a commitment is already
signed by the hotkey, so authorship is chain-attested; a 64-byte SR25519
signature would also consume half the budget to prove something the chain
proves for free.

The encoding is canonical: one commitment has exactly one valid encoding. Mode
``t`` carrying a value that mode ``u`` could have compacted is rejected on
decode, so two validators reading the same commitment cannot disagree about
what a re-encode should look like.
"""

from __future__ import annotations

import base64
import binascii
import re
import uuid
from dataclasses import dataclass
from typing import Any, Final

from prometheon.canonical.integrity import is_valid_revision
from prometheon.chain.subtensor import parse_extrinsic_response
from prometheon.errors import CommitmentDecodeError, CommitmentError

#: The largest ``Raw`` variant substrate will store for a commitment.
COMMITMENT_MAX_BYTES: Final[int] = 128

_MAGIC: Final[str] = "P2"
_FORMAT_VERSION: Final[str] = "1"

#: ``chute_id`` compacted from a canonical UUID to 16 raw bytes.
_MODE_UUID: Final[str] = "u"
#: ``chute_id`` kept as literal text, terminated by :data:`_TERMINATOR`.
_MODE_TEXT: Final[str] = "t"

_HEADER_CHARS: Final[int] = 4
_REVISION_CHARS: Final[int] = 27
_UUID_CHARS: Final[int] = 22
_REVISION_BYTES: Final[int] = 20
_UUID_BYTES: Final[int] = 16

#: Legal in neither a repo id, a chute id, nor the base64url alphabet, which is
#: what makes the literal-mode field unambiguously terminated.
_TERMINATOR: Final[str] = ":"

#: Hugging Face allows ``namespace/name`` or a bare canonical name, each
#: starting alphanumeric. Anything else is not a repo we could fetch anyway.
_HF_REPO_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)

#: Chutes deployment ids are UUIDs. The class is kept slightly wider than that
#: so a platform id-scheme change does not need a format bump. It still
#: excludes the terminator.
_CHUTE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@dataclass(frozen=True)
class ModelCommitment:
    """The three values a miner freezes on chain for a cycle.

    Validated on construction, so an instance in hand is always publishable.
    Every rejection happens at the boundary rather than at submission time,
    when the miner has already lost the day.
    """

    hf_repo: str
    hf_revision: str
    chute_id: str

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
        if not _CHUTE_ID_RE.match(self.chute_id or ""):
            raise CommitmentError(
                f"chute_id must be 1-64 characters of [A-Za-z0-9._-], got {self.chute_id!r}"
            )


def encode_commitment(commitment: ModelCommitment) -> str:
    """Encode ``commitment`` to the ASCII payload written on chain.

    Raises :class:`CommitmentError` if the result would exceed the chain's byte
    budget. Given the fixed costs above, that only happens for an unusually
    long repo id.
    """
    revision_field = _b64url_encode(bytes.fromhex(commitment.hf_revision))
    uuid_bytes = _compactable_uuid_bytes(commitment.chute_id)
    if uuid_bytes is None:
        mode, chute_field = _MODE_TEXT, commitment.chute_id + _TERMINATOR
    else:
        mode, chute_field = _MODE_UUID, _b64url_encode(uuid_bytes)

    text = f"{_MAGIC}{_FORMAT_VERSION}{mode}{revision_field}{chute_field}{commitment.hf_repo}"
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
    if version != _FORMAT_VERSION:
        raise CommitmentDecodeError(
            f"commitment format version {version!r} is not readable by this build "
            f"(which reads {_FORMAT_VERSION!r}); upgrade the validator"
        )

    mode = text[3]
    revision = _b64url_decode(
        text[_HEADER_CHARS : _HEADER_CHARS + _REVISION_CHARS],
        expected_bytes=_REVISION_BYTES,
        field="hf_revision",
    ).hex()
    rest = text[_HEADER_CHARS + _REVISION_CHARS :]

    if mode == _MODE_UUID:
        chute_id, hf_repo = _decode_uuid_mode(rest)
    elif mode == _MODE_TEXT:
        chute_id, hf_repo = _decode_text_mode(rest)
    else:
        raise CommitmentDecodeError(f"unknown chute_id encoding mode {mode!r}")

    if not hf_repo:
        raise CommitmentDecodeError("commitment carries no hf_repo")

    try:
        return ModelCommitment(hf_repo=hf_repo, hf_revision=revision, chute_id=chute_id)
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


def _decode_uuid_mode(rest: str) -> tuple[str, str]:
    if len(rest) < _UUID_CHARS:
        raise CommitmentDecodeError(
            f"commitment is truncated: compact chute_id needs {_UUID_CHARS} characters, "
            f"found {len(rest)}"
        )
    raw = _b64url_decode(rest[:_UUID_CHARS], expected_bytes=_UUID_BYTES, field="chute_id")
    return str(uuid.UUID(bytes=raw)), rest[_UUID_CHARS:]


def _decode_text_mode(rest: str) -> tuple[str, str]:
    chute_id, separator, hf_repo = rest.partition(_TERMINATOR)
    if not separator:
        raise CommitmentDecodeError(f"literal chute_id is not terminated by {_TERMINATOR!r}")
    if _compactable_uuid_bytes(chute_id) is not None:
        # Rejecting this keeps one commitment to one encoding. Without it two
        # validators could re-encode the same on-chain value differently.
        raise CommitmentDecodeError(
            f"chute_id {chute_id!r} is a canonical UUID and must use the compact "
            f"{_MODE_UUID!r} mode"
        )
    return chute_id, hf_repo


def _compactable_uuid_bytes(chute_id: str) -> bytes | None:
    """The 16 raw bytes of ``chute_id``, or ``None`` if it must stay literal.

    Only the canonical lowercase hyphenated spelling compacts. ``uuid.UUID``
    also parses braced, uppercase, and unhyphenated forms, but those would
    decode back to a *different* string than the miner committed, and the
    decoded string is what gets compared against the deployed wrapper.
    """
    try:
        parsed = uuid.UUID(chute_id)
    except (AttributeError, TypeError, ValueError):
        return None
    if str(parsed) != chute_id:
        return None
    return parsed.bytes


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
