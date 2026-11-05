"""Model eligibility: what a validator verifies for itself before scoring.

The hub's signed epoch says which models were frozen for the day; every claim
about what those models are is re-derived from the public Hugging Face API at
the frozen revision. Two validators running this module against the same epoch
reach the same verdicts, so the model pool stays auditable.
"""

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
