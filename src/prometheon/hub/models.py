"""Wire types for the hub: the signed epoch, the results record, credits.

The hub is one small owner-run service. Each closed UTC day it publishes a
**signed epoch**: the miners registered at a frozen block, the frozen model
commitments with their payment status, the content policy's hash, and the hash
and line count of the day's two content files. A validator fetches it, checks
the signature against a key pinned in its own config, downloads the files the
epoch names, checks their hashes, and scores from nothing else.

``docs/hub-contract.md`` is the prose form of these models, and the decisions
behind every field are recorded with the hub's engineer in the shared wire
notes. Where the two disagree, these models are what runs.

**The epoch is read leniently, and that is safe.** Its signature is verified
over the exact bytes received, before any model sees them, so a field this
client does not know is still covered by the signature. Ignoring it changes
nothing the client relies on, while refusing it would turn every additive hub
change into a subnet-wide outage. A change that matters to scoring bumps the
``schema`` string instead, and that *is* refused.

Every instant is an integer. Canonical JSON refuses floats, and the epoch and
the results record are both signed over canonical JSON.
"""

from __future__ import annotations

import datetime as dt
import re
from enum import Enum
from typing import Any, ClassVar, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from prometheon.revision import is_valid_revision

#: Route prefix, relative to the configured hub base URL.
API_PREFIX: Final[str] = "api/v1"

#: The epoch format this client implements.
EPOCH_SCHEMA: Final[str] = "prometheon-epoch/1"

#: Longest body a feed line may carry. BitFan's own limit is far lower; this is
#: the ceiling the evaluation engine and the labeller are sized for.
MAX_CONTENT_CHARS: Final[int] = 32_000

#: Longest item id. Ids reach logs, the labeller and the corpus order key.
MAX_ITEM_ID_CHARS: Final[int] = 128

_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
SS58_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")


def is_ss58(value: object) -> bool:
    return isinstance(value, str) and bool(SS58_RE.match(value))


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX64_RE.match(value))


class HubErrorCode(str, Enum):
    """Every ``error.code`` the hub may return.

    The strings are the v2 contract's, unchanged: the hub kept them, and
    clients key off them rather than off HTTP status.
    """

    BAD_REQUEST = "db.bad_request"
    BAD_CURSOR = "db.bad_cursor"
    AUTH_MALFORMED = "db.auth_malformed"
    AUTH_INVALID = "db.auth_invalid"
    AUTH_EXPIRED = "db.auth_expired"
    AUTH_REPLAYED = "db.auth_replayed"
    NOT_AUTHORIZED = "db.not_authorized"
    EMBARGOED = "db.embargoed"
    NOT_FOUND = "db.not_found"
    SNAPSHOT_NOT_READY = "db.snapshot_not_ready"
    DUPLICATE_EVALUATION = "db.duplicate_evaluation"
    INVALID_PAYLOAD = "db.invalid_payload"
    RATE_LIMITED = "db.rate_limited"
    INTERNAL = "db.internal"


class ErrorDetail(BaseModel):
    """Body of any non-2xx response. Unknown keys are kept out, not refused."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(default="", max_length=2000)
    retry_after_seconds: int | None = Field(default=None, ge=0)
    available_at: int | None = Field(default=None, ge=0)


class ErrorResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    error: ErrorDetail


# ---------------------------------------------------------------------------
# The epoch
# ---------------------------------------------------------------------------


class _Lenient(BaseModel):
    """Signed hub data: unknown fields ignored, see the module docstring."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore", frozen=True, populate_by_name=True
    )


class Payment(str, Enum):
    """Whether a frozen model's submission fee is covered.

    Validators use the published value and never re-derive it: it depends on
    the credit ledger's history, which they do not hold.
    """

    #: Covered by a credit. Evaluated if its hotkey is eligible.
    PAID = "paid"
    #: Nobody could pay for it at the freeze. Listed, not evaluated.
    UNPAID = "unpaid"
    #: Another hotkey holds this fingerprint. Listed, not evaluated.
    DUPLICATE = "duplicate"


class EpochPolicy(_Lenient):
    version: str = Field(min_length=1, max_length=64)
    #: sha256 of the bytes of ``content_policy.md``.
    sha256: str

    @field_validator("sha256")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not is_hex64(value):
            raise ValueError("policy.sha256 must be 64 lowercase hex characters")
        return value


class EpochMiner(_Lenient):
    """A neuron registered at the frozen block.

    Every registered neuron is listed, not only those with a group, because
    self-exclusion needs every hotkey's coldkey.
    """

    hotkey: str
    uid: StrictInt = Field(ge=0)
    coldkey: str
    verified_active_members: StrictInt = Field(ge=0)
    #: The hub's view. A validator also applies its own configured floor; see
    #: :func:`prometheon.hub.epoch.is_eligible`.
    eligible: StrictBool

    @field_validator("hotkey", "coldkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not is_ss58(value):
            raise ValueError(f"not an SS58 address: {value!r}")
        return value


class EpochModel(_Lenient):
    """A model commitment as frozen for the day."""

    hotkey: str
    repo: str = Field(min_length=1, max_length=256)
    revision: str
    #: The first block at which this hotkey committed this fingerprint, so
    #: re-committing an unchanged model never loses an "earliest commit" contest.
    commit_block: StrictInt = Field(ge=0)
    #: sha256 over the sorted Hugging Face LFS sha256s of every safetensors file.
    fingerprint: str
    payment: Payment

    @field_validator("hotkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not is_ss58(value):
            raise ValueError(f"not an SS58 address: {value!r}")
        return value

    @field_validator("revision")
    @classmethod
    def _revision(cls, value: str) -> str:
        if not is_valid_revision(value):
            raise ValueError("revision must be a 40-character lowercase hex commit SHA")
        return value

    @field_validator("fingerprint")
    @classmethod
    def _fingerprint(cls, value: str) -> str:
        if not is_hex64(value):
            raise ValueError("fingerprint must be 64 lowercase hex characters")
        return value


class EpochFile(_Lenient):
    #: sha256 of the gzip bytes, exactly as stored and downloaded.
    sha256: str
    #: Number of JSONL lines in the decompressed file.
    count: StrictInt = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not is_hex64(value):
            raise ValueError("file sha256 must be 64 lowercase hex characters")
        return value


class EpochFiles(_Lenient):
    test: EpochFile
    production: EpochFile


class Epoch(_Lenient):
    """One day, frozen and signed by the hub."""

    schema_name: str = Field(alias="schema")
    netuid: StrictInt = Field(ge=0)
    day: dt.date
    frozen_block: StrictInt = Field(ge=0)
    scoring_version: str = Field(min_length=1, max_length=64)
    policy: EpochPolicy
    miners: tuple[EpochMiner, ...]
    models: tuple[EpochModel, ...]
    files: EpochFiles
    signature: str = Field(min_length=1)
    key_id: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _consistent(self) -> Epoch:
        if self.schema_name != EPOCH_SCHEMA:
            raise ValueError(
                f"epoch schema is {self.schema_name!r}; this client implements {EPOCH_SCHEMA!r}"
            )
        hotkeys = [miner.hotkey for miner in self.miners]
        if len(set(hotkeys)) != len(hotkeys):
            raise ValueError("a hotkey appears twice in the epoch's miner list")
        uids = [miner.uid for miner in self.miners]
        if len(set(uids)) != len(uids):
            raise ValueError("a uid appears twice in the epoch's miner list")
        model_hotkeys = [model.hotkey for model in self.models]
        if len(set(model_hotkeys)) != len(model_hotkeys):
            raise ValueError("a hotkey has two frozen models; a hotkey commits one at a time")
        return self

    def miner(self, hotkey: str) -> EpochMiner | None:
        for miner in self.miners:
            if miner.hotkey == hotkey:
                return miner
        return None


class EpochUrls(_Lenient):
    """Presigned download URLs, valid for an hour."""

    test: str = Field(min_length=1)
    production: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# The results record
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ModelStatus(str, Enum):
    """Why a miner did or did not get a model score."""

    #: No model in the frozen list.
    NOT_COMMITTED = "not_committed"
    #: Listed, but its submission fee is not covered.
    UNPAID = "unpaid"
    #: Listed, but another hotkey holds the same weights.
    DUPLICATE = "duplicate"
    #: Listed and paid, but the hotkey is below the eligibility floor.
    INELIGIBLE = "ineligible"
    #: Listed and paid, but the repository failed a validator's checks.
    INVALID = "invalid"
    #: Evaluated.
    EVALUATED = "evaluated"
    #: Verified, but could not be downloaded or run.
    UNREACHABLE = "unreachable"


class MinerResult(_Strict):
    """One miner's line in a validator's published record.

    Every number is an integer: the record is signed over canonical JSON, which
    refuses floats.
    """

    hotkey: str = Field(min_length=1, max_length=64)
    uid: int = Field(ge=0)
    #: Units this hotkey earned from the two pools. **Never includes burn**:
    #: burn travels in ``EvaluationSubmission.burned_weight``, so the rows plus
    #: the burn reproduce the vector exactly even when the owner also earns.
    weight: int = Field(ge=0)

    #: ``S``: the miner's test items left after near-duplicate removal.
    dataset_submitted_count: int = Field(ge=0)
    #: ``V_pos + V_neg``.
    dataset_valid_count: int = Field(ge=0)
    #: ``V_pos``: claimed violating, confirmed violating.
    dataset_violating_confirmed: int = Field(ge=0)
    #: ``V_neg``: claimed compliant, confirmed compliant.
    dataset_compliant_confirmed: int = Field(ge=0)
    #: ``2 * min(V_pos, V_neg) * (V_pos + V_neg) / S``, times 1e6, floored.
    dataset_score_micro: int = Field(ge=0)

    model_status: ModelStatus
    model_items_scored: int = Field(ge=0)
    model_correct_count: int = Field(ge=0)
    raw_accuracy_bp: int = Field(ge=0, le=10_000)
    mean_total_tokens_milli: int = Field(ge=0)
    efficiency_penalty_bp: int = Field(ge=0, le=10_000)
    model_score_bp: int = Field(ge=0, le=10_000)
    model_rank: int | None = Field(default=None, ge=1)


class EvaluationSubmission(_Strict):
    """A validator's signed account of one day.

    Signed under ``PROMETHEON_V2_EVALUATION`` over every field except the
    signature, which is exactly what the hub verifies: every top-level field of
    the received JSON except ``signature``. :meth:`signing_payload` is built by
    hand, and a test pins that it equals the serialised record minus the
    signature, so a field added here cannot silently fall outside it.
    """

    date: dt.date
    netuid: int = Field(ge=0)
    validator_hotkey: str = Field(min_length=1, max_length=64)
    scoring_version: str = Field(min_length=1, max_length=64)
    policy_version: str = Field(min_length=1, max_length=64)
    #: sha256 of the exact bytes the epoch signature covers. Two validators
    #: that disagree here were not scoring the same day.
    epoch_content_hash: str = Field(min_length=64, max_length=64)
    #: Hash over the labelled corpus this validator evaluated.
    corpus_content_hash: str = Field(min_length=64, max_length=64)
    labelled_test_count: int = Field(ge=0)
    labelled_production_count: int = Field(ge=0)
    results: tuple[MinerResult, ...]
    #: Units sent to the subnet owner hotkey as burn: unclaimed shares, plus the
    #: units of any hotkey that had deregistered by the time weights were sent.
    burned_weight: int = Field(ge=0)
    started_at: int = Field(ge=0)
    completed_at: int = Field(ge=0)
    #: ``0x`` + 128 hex. Empty only on an unsigned draft.
    signature: str = Field(default="", max_length=130)

    @field_validator("epoch_content_hash", "corpus_content_hash")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not is_hex64(value):
            raise ValueError("content hashes must be 64 lowercase hex characters")
        return value

    def signing_payload(self) -> dict[str, Any]:
        return {
            "burned_weight": self.burned_weight,
            "completed_at": self.completed_at,
            "corpus_content_hash": self.corpus_content_hash,
            "date": self.date.isoformat(),
            "epoch_content_hash": self.epoch_content_hash,
            "labelled_production_count": self.labelled_production_count,
            "labelled_test_count": self.labelled_test_count,
            "netuid": self.netuid,
            "policy_version": self.policy_version,
            "results": [_result_payload(result) for result in self.results],
            "scoring_version": self.scoring_version,
            "started_at": self.started_at,
            "validator_hotkey": self.validator_hotkey,
        }


def _result_payload(result: MinerResult) -> dict[str, Any]:
    return {
        "dataset_compliant_confirmed": result.dataset_compliant_confirmed,
        "dataset_score_micro": result.dataset_score_micro,
        "dataset_submitted_count": result.dataset_submitted_count,
        "dataset_valid_count": result.dataset_valid_count,
        "dataset_violating_confirmed": result.dataset_violating_confirmed,
        "efficiency_penalty_bp": result.efficiency_penalty_bp,
        "hotkey": result.hotkey,
        "mean_total_tokens_milli": result.mean_total_tokens_milli,
        "model_correct_count": result.model_correct_count,
        "model_items_scored": result.model_items_scored,
        "model_rank": result.model_rank,
        "model_score_bp": result.model_score_bp,
        "model_status": result.model_status.value,
        "raw_accuracy_bp": result.raw_accuracy_bp,
        "uid": result.uid,
        "weight": result.weight,
    }


class EvaluationAccepted(BaseModel):
    """Receipt for a stored record. ``duplicate`` means these exact bytes were
    already stored, which is how a client that lost a response finds out it
    landed."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    evaluation_id: str = Field(min_length=1, max_length=256)
    accepted_at: int = Field(ge=0)
    duplicate: bool = False


# ---------------------------------------------------------------------------
# Credits
# ---------------------------------------------------------------------------


class CreditEntry(BaseModel):
    """One ledger line: a deposit read from chain, or a spend at a freeze."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    kind: str = Field(min_length=1, max_length=32)
    block: int = Field(ge=0)
    extrinsic_hash: str | None = None
    alpha_rao: int | None = None
    credits: int | None = None
    hotkey: str | None = None
    fingerprint: str | None = None


class CreditBalance(BaseModel):
    """A coldkey's submission credits, recomputable by anyone from chain events."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    coldkey: str
    balance: int = Field(ge=0)
    remainder_alpha_rao: int = Field(default=0, ge=0)
    submission_fee_alpha_rao: int | None = Field(default=None, ge=0)
    entries: tuple[CreditEntry, ...] = ()


__all__ = [
    "API_PREFIX",
    "EPOCH_SCHEMA",
    "MAX_CONTENT_CHARS",
    "MAX_ITEM_ID_CHARS",
    "SS58_RE",
    "CreditBalance",
    "CreditEntry",
    "Epoch",
    "EpochFile",
    "EpochFiles",
    "EpochMiner",
    "EpochModel",
    "EpochPolicy",
    "EpochUrls",
    "ErrorDetail",
    "ErrorResponse",
    "EvaluationAccepted",
    "EvaluationSubmission",
    "HubErrorCode",
    "MinerResult",
    "ModelStatus",
    "Payment",
    "is_hex64",
    "is_ss58",
]
