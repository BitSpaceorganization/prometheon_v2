"""Client, wire types, and local fake for the subnet DB layer.

The DB layer is a separate private service that owns everything the subnet
cannot read from chain: platform-sourced test content, production traffic,
which miners are eligible, and the frozen daily commitment set. It is the only
bridge between the platform and the validator cycle.

``docs/db-api-contract.md`` is the normative specification. This package is the
reference client, plus :class:`FakeDbLayer`, an in-memory implementation of the
same contract that makes the whole validator path testable before the real
service exists.
"""

from prometheon.dbclient.auth import (
    AUTH_MAX_CLOCK_SKEW_SECONDS,
    AuthenticatedCaller,
    CallerRole,
    HotkeyDirectory,
    InMemoryNonceStore,
    NonceStore,
    RequestEnvelope,
    StaticHotkeyDirectory,
    build_envelope,
    envelope_headers,
    new_nonce,
    sign_envelope,
    sign_evaluation,
    signed_headers,
    verify_evaluation,
    verify_request,
)
from prometheon.dbclient.client import DaySnapshot, DbClient
from prometheon.dbclient.fake import DaySeed, FakeDbLayer
from prometheon.dbclient.models import (
    API_VERSION,
    DEFAULT_PAGE_LIMIT,
    EMBARGO_DAYS,
    MAX_PAGE_LIMIT,
    PRODUCTION_ITEM_CAP,
    SNAPSHOT_VERSION,
    DbErrorCode,
    EligibleMiner,
    EligibleMinerSet,
    ErrorDetail,
    ErrorResponse,
    EvaluationAccepted,
    EvaluationSubmission,
    MinerResult,
    ModelCommitment,
    ModelCommitmentSet,
    ModelStatus,
    ProductionContentItem,
    ProductionContentPage,
    SnapshotManifest,
    TestContentItem,
    TestContentPage,
    corpus_content_hash,
    snapshot_content_hash,
)

__all__ = [
    "API_VERSION",
    "AUTH_MAX_CLOCK_SKEW_SECONDS",
    "DEFAULT_PAGE_LIMIT",
    "EMBARGO_DAYS",
    "MAX_PAGE_LIMIT",
    "PRODUCTION_ITEM_CAP",
    "SNAPSHOT_VERSION",
    "AuthenticatedCaller",
    "CallerRole",
    "DaySeed",
    "DaySnapshot",
    "DbClient",
    "DbErrorCode",
    "EligibleMiner",
    "EligibleMinerSet",
    "ErrorDetail",
    "ErrorResponse",
    "EvaluationAccepted",
    "EvaluationSubmission",
    "FakeDbLayer",
    "HotkeyDirectory",
    "InMemoryNonceStore",
    "MinerResult",
    "ModelCommitment",
    "ModelCommitmentSet",
    "ModelStatus",
    "NonceStore",
    "ProductionContentItem",
    "ProductionContentPage",
    "RequestEnvelope",
    "SnapshotManifest",
    "StaticHotkeyDirectory",
    "TestContentItem",
    "TestContentPage",
    "build_envelope",
    "corpus_content_hash",
    "envelope_headers",
    "new_nonce",
    "sign_envelope",
    "sign_evaluation",
    "signed_headers",
    "snapshot_content_hash",
    "verify_evaluation",
    "verify_request",
]
