"""Model eligibility: what a validator verifies for itself before scoring.

Nothing here is taken on trust. The DB layer supplies the list of miners to
consider; every claim about what those miners published is re-derived from the
chain commitment and the public Hugging Face API. Two validators running this
module against the same block reach the same verdicts, so the model half of
emission stays auditable.
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
