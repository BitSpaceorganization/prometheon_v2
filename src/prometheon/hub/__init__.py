"""The hub: the signed daily epoch a validator scores from, and where results go.

The hub is one small owner-run service. It reads BitFan's daily contribution
feed and an index of the chain, freezes each closed UTC day into a signed
**epoch**, stores the day's two content files, and keeps the records validators
publish. ``docs/hub-contract.md`` is the normative description; this package is
the reference client, plus :class:`~prometheon.hub.fake.FakeHub`, an in-memory
hub the whole validator path is tested against.
"""

from prometheon.hub.client import DayData, EpochBundle, HubClient, VerifiedEpoch
from prometheon.hub.epoch import (
    check_epoch_is_for,
    eligible_miners,
    epoch_content_hash,
    epoch_signing_bytes,
    is_eligible,
    paid_models,
    verify_epoch,
)
from prometheon.hub.feed import DayFeed, ProductionLine, RejectedLine, TestLine
from prometheon.hub.models import (
    EPOCH_SCHEMA,
    CreditBalance,
    Epoch,
    EpochMiner,
    EpochModel,
    EvaluationAccepted,
    EvaluationSubmission,
    MinerResult,
    ModelStatus,
    Payment,
)

__all__ = [
    "EPOCH_SCHEMA",
    "CreditBalance",
    "DayData",
    "DayFeed",
    "Epoch",
    "EpochBundle",
    "EpochMiner",
    "EpochModel",
    "EvaluationAccepted",
    "EvaluationSubmission",
    "HubClient",
    "MinerResult",
    "ModelStatus",
    "Payment",
    "ProductionLine",
    "RejectedLine",
    "TestLine",
    "VerifiedEpoch",
    "check_epoch_is_for",
    "eligible_miners",
    "epoch_content_hash",
    "epoch_signing_bytes",
    "is_eligible",
    "paid_models",
    "verify_epoch",
]
