"""Model eligibility: what a validator verifies for itself before scoring.

Nothing here is taken on trust. The DB layer supplies the list of miners to
consider; every claim about what those miners published and deployed is
re-derived from the chain commitment, the public Hugging Face API, and the
Chutes API. Two validators running this module against the same block reach the
same verdicts, so the model half of emission stays auditable.
"""

from prometheon.registry.chutes import (
    ChuteInfo,
    ChutesClient,
    require_hot,
)
from prometheon.registry.huggingface import (
    HuggingFaceClient,
    RepoSnapshot,
    file_allowed,
    file_manifest,
)
from prometheon.registry.validation import (
    InvalidReason,
    MinerEntry,
    MinerValidation,
    ModelRegistry,
    eligible_miners,
)

__all__ = [
    "ChuteInfo",
    "ChutesClient",
    "HuggingFaceClient",
    "InvalidReason",
    "MinerEntry",
    "MinerValidation",
    "ModelRegistry",
    "RepoSnapshot",
    "eligible_miners",
    "file_allowed",
    "file_manifest",
    "require_hot",
]
