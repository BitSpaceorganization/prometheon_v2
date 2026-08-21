"""Wire types for the subnet DB layer, and the snapshot content hash.

These models *are* the contract. ``docs/db-api-contract.md`` is their prose
form; if the two ever disagree, the models are what runs and the document is
the bug.

Two decisions constrain every field below.

**Every instant is an integer count of seconds since the Unix epoch, UTC.** Not
an ISO-8601 string, not a float. Instants end up inside signed payloads and
inside the snapshot content hash, and canonical JSON rejects floats because two
implementations may format them differently. Integers remove the question.
Calendar days stay ``YYYY-MM-DD`` because a day is not an instant.

**Responses are validated with ``extra="forbid"``.** A field nobody expected
means the bytes a validator hashed are not the bytes it was served, and the
content hash exists so that independent validators agree on the corpus. An
added field is therefore a breaking change and needs a new API version rather
than a silent rollout. The one exception is the error body, which is allowed to
grow diagnostics: refusing to parse an error is how a client turns a readable
failure into an unreadable one.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import re
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from prometheon.revision import is_valid_revision
from prometheon.security.canonical_json import to_canonical_bytes

#: URL segment for this revision of the contract.
API_VERSION: Final[str] = "v2"

#: Stamped into every manifest and folded into the content hash. A change here
#: changes every hash, which is the point: it makes an incompatible snapshot
#: build impossible to mistake for a compatible one.
SNAPSHOT_VERSION: Final[str] = "prometheon-snapshot/2"

#: Hard ceiling on production content in a single day's snapshot. Validators
#: size their labelling budget against it, so the DB layer samples down to this
#: rather than serving whatever the platform produced.
PRODUCTION_ITEM_CAP: Final[int] = 10_000

#: Page sizes. A server may serve fewer items than asked for; it may never
#: serve more, and a client must follow ``next_cursor`` rather than assume a
#: page size.
DEFAULT_PAGE_LIMIT: Final[int] = 500
MAX_PAGE_LIMIT: Final[int] = 1_000

#: Days a miner must wait before it may read a day's content.
EMBARGO_DAYS: Final[int] = 2

#: Longest content the DB layer will store or serve for one item. Matches the
#: evaluation engine's own limit, so a validator can never build a corpus item
#: that the engine would reject as oversized.
MAX_CONTENT_CHARS: Final[int] = 32_000

_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_SS58_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")

#: Cursors are base64url without padding. Constraining the alphabet is what
#: lets the request signature cover the query string without either side having
#: to agree on a percent-encoding.
CURSOR_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,512}$")


class DbErrorCode(str, Enum):
    """Every ``error.code`` the DB layer may return.

    Clients key off these strings, not off HTTP status, because a status is too
    coarse to act on: 403 is both "you are not a validator" and "come back in
    two days", and those have different remediations.
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


class _Strict(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ErrorDetail(BaseModel):
    """Body of any non-2xx response.

    Unknown keys are ignored rather than rejected: an error that cannot be
    parsed is an error that cannot be reported, and diagnostics are exactly the
    thing a server will want to add later.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(default="", max_length=2000)
    #: Set on ``db.snapshot_not_ready`` and ``db.rate_limited``.
    retry_after_seconds: int | None = Field(default=None, ge=0)
    #: Set on ``db.embargoed``: the instant the caller may read this day.
    available_at: int | None = Field(default=None, ge=0)


class ErrorResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


class TestContentItem(_Strict):
    """One piece of miner-sourced content that is *meant* to violate policy.

    Attributed, because attribution is what the dataset half of the reward is
    computed from and what self-exclusion needs: a miner is never scored on
    items its own users contributed.
    """

    id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    author_hotkey: str = Field(min_length=1, max_length=64)
    submitted_at: int = Field(ge=0)

    @field_validator("author_hotkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not _SS58_RE.match(value):
            raise ValueError(f"author_hotkey is not an SS58 address: {value!r}")
        return value


class ProductionContentItem(_Strict):
    """One piece of real platform traffic.

    Carries **no author field of any kind**, and ``extra="forbid"`` is what
    enforces that: a server that starts attributing production content makes
    every client raise on the next response rather than quietly handing
    validators a way to tell whose traffic they are scoring. Production content
    is where the negatives come from, so attribution here would be a direct
    lever on scoring.
    """

    id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    observed_at: int = Field(ge=0)


class EligibleMiner(_Strict):
    """A miner the platform considers eligible to be scored on day N.

    Carries no submitted/valid counts. Those are the numerator and denominator
    of the dataset reward, and a validator derives them itself by labelling the
    attributed test content. Asking the DB layer for them would move ground
    truth off the validator and into the service being audited.
    """

    hotkey: str = Field(min_length=1, max_length=64)
    qualified_user_count: int = Field(ge=0)

    @field_validator("hotkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not _SS58_RE.match(value):
            raise ValueError(f"hotkey is not an SS58 address: {value!r}")
        return value


class ModelCommitment(_Strict):
    """A miner's on-chain model commitment, frozen at 00:00 UTC of the day.

    Mirrored here rather than read from chain by every validator so that all
    validators evaluate the *same* set: a commitment written at 03:59 must not
    change what one validator scores and another does not.

    **The deployment id is gone, and its absence is enforced.** Responses are
    validated with ``extra="forbid"``, so a DB layer still sending it makes the
    snapshot unparseable rather than being quietly ignored. That is deliberate:
    the field named a deployment validators no longer call, and a snapshot
    carrying one describes a submission under rules that no longer apply.
    """

    hotkey: str = Field(min_length=1, max_length=64)
    hf_repo: str = Field(min_length=1, max_length=256)
    revision_sha: str = Field(min_length=40, max_length=40)
    committed_at: int = Field(ge=0)
    block: int = Field(ge=0)

    @field_validator("revision_sha")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not is_valid_revision(value):
            raise ValueError("revision_sha must be a 40-character lowercase hex commit SHA")
        return value

    @field_validator("hotkey")
    @classmethod
    def _ss58(cls, value: str) -> str:
        if not _SS58_RE.match(value):
            raise ValueError(f"hotkey is not an SS58 address: {value!r}")
        return value


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SnapshotManifest(_Strict):
    """What day N contains, and one hash that fixes all of it."""

    date: dt.date
    snapshot_version: str = Field(min_length=1, max_length=64)
    built_at: int = Field(ge=0)
    test_item_count: int = Field(ge=0)
    production_item_count: int = Field(ge=0, le=PRODUCTION_ITEM_CAP)
    eligible_miner_count: int = Field(ge=0)
    model_commitment_count: int = Field(ge=0)
    content_hash: str = Field(min_length=64, max_length=64)

    @field_validator("content_hash")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not _HEX64_RE.match(value):
            raise ValueError("content_hash must be 64 lowercase hex characters")
        return value


class TestContentPage(_Strict):
    date: dt.date
    items: tuple[TestContentItem, ...]
    next_cursor: str | None = None
    total_count: int = Field(ge=0)


class ProductionContentPage(_Strict):
    date: dt.date
    items: tuple[ProductionContentItem, ...]
    next_cursor: str | None = None
    total_count: int = Field(ge=0, le=PRODUCTION_ITEM_CAP)


class EligibleMinerSet(_Strict):
    date: dt.date
    miners: tuple[EligibleMiner, ...]


class ModelCommitmentSet(_Strict):
    date: dt.date
    #: 00:00 UTC of ``date``. Stated rather than implied so a validator can
    #: assert the freeze it was promised actually happened.
    frozen_at: int = Field(ge=0)
    commitments: tuple[ModelCommitment, ...]


# ---------------------------------------------------------------------------
# Evaluation submission
# ---------------------------------------------------------------------------


class ModelStatus(str, Enum):
    """Why a miner did or did not get a model score."""

    #: No on-chain commitment in the frozen set.
    NOT_COMMITTED = "not_committed"
    #: Committed, but the deployment failed registry verification.
    INVALID = "invalid"
    #: Committed, verified, and served the corpus.
    EVALUATED = "evaluated"
    #: Committed and verified, but unreachable or non-conforming at call time.
    UNREACHABLE = "unreachable"


class MinerResult(_Strict):
    """One miner's line in a validator's published result.

    Every number is an integer. Ratios are carried as basis points or as
    fixed-point micros, named so in the field, because this payload is signed
    and canonical JSON refuses floats. ``scoring`` owns what the numbers mean;
    this module only fixes how they travel.
    """

    hotkey: str = Field(min_length=1, max_length=64)
    uid: int = Field(ge=0)
    #: The integer weight actually submitted for this miner.
    weight: int = Field(ge=0)

    dataset_submitted_count: int = Field(ge=0)
    dataset_valid_count: int = Field(ge=0)
    #: ``V^2/S`` scaled by 1e6 and floored.
    dataset_score_micro: int = Field(ge=0)

    model_status: ModelStatus
    model_items_scored: int = Field(ge=0)
    model_correct_count: int = Field(ge=0)
    raw_accuracy_bp: int = Field(ge=0, le=10_000)
    #: Mean total tokens per item, scaled by 1000. Zero when not evaluated.
    mean_total_tokens_milli: int = Field(ge=0)
    #: ``lambda * t`` in basis points; zero when the CV guard switched it off.
    efficiency_penalty_bp: int = Field(ge=0, le=10_000)
    model_score_bp: int = Field(ge=0, le=10_000)
    #: 1-based, or ``None`` when this miner earned no model rank.
    model_rank: int | None = Field(default=None, ge=1)


class EvaluationSubmission(_Strict):
    """A validator's signed, reproducible account of one cycle.

    Signed under ``DOMAIN_EVALUATION`` over :meth:`signing_payload`, which is
    every field except the signature itself. Anyone holding the same day's
    snapshot can recompute the numbers and check the signature, so the subnet
    is auditable without trusting the DB layer.
    """

    date: dt.date
    netuid: int = Field(ge=0)
    validator_hotkey: str = Field(min_length=1, max_length=64)
    scoring_version: str = Field(min_length=1, max_length=64)
    #: The ``Version:`` line of the ``content_policy.md`` used for labelling.
    policy_version: str = Field(min_length=1, max_length=64)
    #: The manifest hash this cycle consumed. Two validators that disagree here
    #: were not scoring the same day, and that is worth seeing in the record.
    snapshot_content_hash: str = Field(min_length=64, max_length=64)
    #: Hash over the labelled corpus actually evaluated, per
    #: :func:`corpus_content_hash`.
    corpus_content_hash: str = Field(min_length=64, max_length=64)
    labelled_test_count: int = Field(ge=0)
    labelled_production_count: int = Field(ge=0)
    results: tuple[MinerResult, ...]
    #: Weight sent to the subnet owner hotkey when a pool had no eligible
    #: recipients and burned.
    burned_weight: int = Field(ge=0)
    started_at: int = Field(ge=0)
    completed_at: int = Field(ge=0)
    #: ``0x`` + 128 hex. Empty only on an unsigned draft, which a server rejects.
    signature: str = Field(default="", max_length=130)

    @field_validator("snapshot_content_hash", "corpus_content_hash")
    @classmethod
    def _hex(cls, value: str) -> str:
        if not _HEX64_RE.match(value):
            raise ValueError("content hashes must be 64 lowercase hex characters")
        return value

    def signing_payload(self) -> dict[str, Any]:
        """The exact object signed under ``DOMAIN_EVALUATION``.

        Built by hand rather than by dumping the model, so that adding a field
        to the class can never silently change what a signature covers.
        """
        return {
            "burned_weight": self.burned_weight,
            "completed_at": self.completed_at,
            "corpus_content_hash": self.corpus_content_hash,
            "date": self.date.isoformat(),
            "labelled_production_count": self.labelled_production_count,
            "labelled_test_count": self.labelled_test_count,
            "netuid": self.netuid,
            "policy_version": self.policy_version,
            "results": [_result_payload(result) for result in self.results],
            "scoring_version": self.scoring_version,
            "snapshot_content_hash": self.snapshot_content_hash,
            "started_at": self.started_at,
            "validator_hotkey": self.validator_hotkey,
        }


def _result_payload(result: MinerResult) -> dict[str, Any]:
    return {
        "dataset_score_micro": result.dataset_score_micro,
        "dataset_submitted_count": result.dataset_submitted_count,
        "dataset_valid_count": result.dataset_valid_count,
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


class EvaluationAccepted(_Strict):
    """Receipt for a stored evaluation."""

    evaluation_id: str = Field(min_length=1, max_length=128)
    accepted_at: int = Field(ge=0)
    #: True when this exact submission was already stored. Resubmitting a
    #: byte-identical payload succeeds; treating it as a conflict would leave a
    #: client that lost a response no safe way to find out whether it landed.
    duplicate: bool = False


# ---------------------------------------------------------------------------
# The snapshot content hash
# ---------------------------------------------------------------------------


def _member_digest(payload: dict[str, Any]) -> bytes:
    return hashlib.sha256(to_canonical_bytes(payload)).digest()


def test_item_digest(item: TestContentItem) -> bytes:
    return _member_digest(
        {
            "author_hotkey": item.author_hotkey,
            "content": item.content,
            "id": item.id,
            "submitted_at": item.submitted_at,
        }
    )


def production_item_digest(item: ProductionContentItem) -> bytes:
    return _member_digest({"content": item.content, "id": item.id, "observed_at": item.observed_at})


def eligible_miner_digest(miner: EligibleMiner) -> bytes:
    return _member_digest(
        {"hotkey": miner.hotkey, "qualified_user_count": miner.qualified_user_count}
    )


def model_commitment_digest(commitment: ModelCommitment) -> bytes:
    return _member_digest(
        {
            "block": commitment.block,
            "committed_at": commitment.committed_at,
            "hf_repo": commitment.hf_repo,
            "hotkey": commitment.hotkey,
            "revision_sha": commitment.revision_sha,
        }
    )


def collection_hash(digests: Iterable[bytes]) -> str:
    """Order-independent hash of a collection of member digests.

    Sorting the digests before folding them is what makes the result
    independent of pagination order, so a server may reshard or reorder pages
    without changing the day's hash. A changed, added, or dropped item changes
    it immediately. Sorting keeps duplicates (they appear twice), so
    multiplicity is still covered.
    """
    joined = b"".join(sorted(digests))
    return hashlib.sha256(joined).hexdigest()


def snapshot_content_hash(
    *,
    date: dt.date,
    test_items: Sequence[TestContentItem],
    production_items: Sequence[ProductionContentItem],
    eligible_miners: Sequence[EligibleMiner],
    model_commitments: Sequence[ModelCommitment],
) -> str:
    """The one hash that fixes an entire day.

    A validator recomputes this from everything it downloaded and compares it
    to the manifest. That check is the only thing standing between the subnet
    and a DB layer that serves one corpus to one validator and a different one
    to another, which would split the field's weights while looking, from any
    single validator's side, entirely normal.
    """
    return hashlib.sha256(
        to_canonical_bytes(
            {
                "counts": {
                    "eligible_miners": len(eligible_miners),
                    "models": len(model_commitments),
                    "production": len(production_items),
                    "test": len(test_items),
                },
                "date": date.isoformat(),
                "eligible_miners": collection_hash(
                    eligible_miner_digest(miner) for miner in eligible_miners
                ),
                "models": collection_hash(
                    model_commitment_digest(commitment) for commitment in model_commitments
                ),
                "production": collection_hash(
                    production_item_digest(item) for item in production_items
                ),
                "snapshot_version": SNAPSHOT_VERSION,
                "test": collection_hash(test_item_digest(item) for item in test_items),
            }
        )
    ).hexdigest()


def corpus_content_hash(*, date: dt.date, labelled: Sequence[tuple[str, bool]]) -> str:
    """Hash over the labelled corpus a validator actually evaluated.

    ``labelled`` is ``(item_id, violates)`` pairs. Published alongside the
    snapshot hash because the two answer different questions: the snapshot hash
    says "we were given the same data", this says "we made the same calls out
    of it". A labeller disagreement shows up here and nowhere else.
    """
    digests = (
        _member_digest({"id": item_id, "violates": violates}) for item_id, violates in labelled
    )
    return hashlib.sha256(
        to_canonical_bytes({"date": date.isoformat(), "labelled": collection_hash(digests)})
    ).hexdigest()


__all__ = [
    "API_VERSION",
    "CURSOR_RE",
    "DEFAULT_PAGE_LIMIT",
    "EMBARGO_DAYS",
    "MAX_CONTENT_CHARS",
    "MAX_PAGE_LIMIT",
    "PRODUCTION_ITEM_CAP",
    "SNAPSHOT_VERSION",
    "DbErrorCode",
    "EligibleMiner",
    "EligibleMinerSet",
    "ErrorDetail",
    "ErrorResponse",
    "EvaluationAccepted",
    "EvaluationSubmission",
    "MinerResult",
    "ModelCommitment",
    "ModelCommitmentSet",
    "ModelStatus",
    "ProductionContentItem",
    "ProductionContentPage",
    "SnapshotManifest",
    "TestContentItem",
    "TestContentPage",
    "collection_hash",
    "corpus_content_hash",
    "eligible_miner_digest",
    "model_commitment_digest",
    "production_item_digest",
    "snapshot_content_hash",
    "test_item_digest",
]
