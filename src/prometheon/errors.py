"""The error taxonomy.

Every failure an operator can see carries a stable ``code``. Codes are the
contract: monitoring, the troubleshooting tables in ``docs/``, and the renderer
all key off them, and they are never reworded without a doc change.

The rule that shaped this module: an exception that reaches an operator must
say what to do about it, not only what went wrong. Anything raised here should
be renderable with a remediation, which is why every class carries a code and
why the generic bases exist rather than bare ``RuntimeError``.
"""

from __future__ import annotations


class PrometheonError(Exception):
    """Base for every subnet-side failure."""

    code: str = "prometheon.error"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(PrometheonError):
    """The TOML config was rejected before anything else ran."""

    code: str = "config.invalid"


# ---------------------------------------------------------------------------
# Security primitives
# ---------------------------------------------------------------------------


class SecurityError(PrometheonError):
    code: str = "security.error"


class CanonicalEncodingError(SecurityError):
    """A payload could not be canonicalised, so it must not be signed."""

    code: str = "security.canonical_encoding"


class SignatureError(SecurityError):
    code: str = "security.signature_error"


class SignatureFormatError(SignatureError):
    code: str = "security.signature_malformed"


class SignatureVerificationError(SignatureError):
    code: str = "security.signature_invalid"


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------


class ChainError(PrometheonError):
    code: str = "chain.error"


class SubtensorError(ChainError):
    """Connect, metagraph sync, or a hyperparameter read failed.

    Distinct from :class:`WeightSubmissionError` because the remediation is
    entirely different: this one is connectivity, that one is policy.
    """

    code: str = "chain.subtensor_error"


class CommitmentError(ChainError):
    """A model commitment could not be written, read, or decoded."""

    code: str = "chain.commitment_error"


class CommitmentDecodeError(CommitmentError):
    """An on-chain commitment exists but is not a Prometheon V2 payload."""

    code: str = "chain.commitment_malformed"


class WeightSubmissionError(ChainError):
    code: str = "chain.weight_submission_error"


class SetWeightsFailedError(WeightSubmissionError):
    """The extrinsic was rejected or reported failure."""

    code: str = "chain.set_weights_failed"


class CommitRevealEnabledError(WeightSubmissionError):
    """The subnet has commit-reveal on; this runtime does not implement it."""

    code: str = "chain.commit_reveal_enabled"


class WeightsVersionMismatchError(WeightSubmissionError):
    code: str = "chain.weights_version_mismatch"


class MechidMissingError(WeightSubmissionError):
    """The SDK cannot name the mechanism these weights apply to.

    Distinct from the base class because the remediation is distinct: this one
    is fixed by upgrading the SDK, not by changing config or waiting out a
    chain state. On a multi-mechanism subnet, weights submitted without a
    mechid land on whatever the chain defaults to and nothing reports it.
    """

    code: str = "chain.mechid_missing"


class DuplicateWeightTargetError(WeightSubmissionError):
    """The same UID appears twice in one weight vector.

    Fatal rather than merged: the chain keeps one of the two entries, so a
    duplicate means some target silently receives a different weight than the
    one scoring computed.
    """

    code: str = "chain.duplicate_weight_target"


# ---------------------------------------------------------------------------
# Registry: model eligibility
# ---------------------------------------------------------------------------


class RegistryError(PrometheonError):
    code: str = "registry.error"


class ManifestViolationError(RegistryError):
    """The Hugging Face repo carries a file outside the permitted manifest."""

    code: str = "registry.manifest_violation"


class EngineHashMismatchError(RegistryError):
    """``miner.py`` is not the canonical engine."""

    code: str = "registry.engine_hash_mismatch"


class WrapperHashMismatchError(RegistryError):
    """The deployed chute source is not the canonical wrapper."""

    code: str = "registry.wrapper_hash_mismatch"


class RevisionFormatError(RegistryError):
    """A revision must be a 40-character hex SHA, never a branch or tag."""

    code: str = "registry.revision_not_sha"


class CommitmentMismatchError(RegistryError):
    """The deployed wrapper declares a repo or revision the chain does not."""

    code: str = "registry.commitment_mismatch"


class ImageMismatchError(RegistryError):
    """The deployment runs an image the subnet did not publish.

    File hashes prove which bytes are on disk; they prove nothing about the
    interpreter that loads them. Pinning the image is what makes "every model
    runs identical code" a fact rather than a description, and the id is the
    only handle the platform exposes.
    """

    code: str = "registry.image_mismatch"


class ChuteNotRunningError(RegistryError):
    code: str = "registry.chute_not_running"


class DuplicateModelError(RegistryError):
    """Another hotkey committed this exact model first."""

    code: str = "registry.duplicate_model"


# ---------------------------------------------------------------------------
# Subnet DB layer
# ---------------------------------------------------------------------------


class DbLayerError(PrometheonError):
    """A call to the subnet DB layer failed.

    Carries the HTTP status and the layer's own error code when the failure
    came from a response, so a caller can distinguish "not released yet" from
    "you are not a validator" without parsing prose.
    """

    code: str = "db.error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class SnapshotNotReadyError(DbLayerError):
    """The day's snapshot has not been built yet. Wait and retry."""

    code: str = "db.snapshot_not_ready"


class EmbargoedError(DbLayerError):
    """A miner asked for a dataset still inside its embargo window."""

    code: str = "db.embargoed"


class NotAuthorizedError(DbLayerError):
    """The hotkey is not a registered validator on this netuid."""

    code: str = "db.not_authorized"


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


class LabellingError(PrometheonError):
    code: str = "labelling.error"


class LabelResponseError(LabellingError):
    """The labeller returned something that is not a usable answer set.

    Raised when ids do not echo, the count is wrong, or a verdict is not a
    boolean. Never repaired by guessing: the batch is split and retried, and
    items that still fail are dropped from the corpus.
    """

    code: str = "labelling.response_invalid"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationError(PrometheonError):
    code: str = "evaluation.error"


class ModelResponseError(EvaluationError):
    """A model's response did not satisfy the wrapper contract.

    Per the scoring rules this is not fatal: the batch is scored as incorrect
    and evaluation continues with the next one.
    """

    code: str = "evaluation.response_invalid"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ScoringError(PrometheonError):
    code: str = "scoring.error"


__all__ = [
    "CanonicalEncodingError",
    "ChainError",
    "ChuteNotRunningError",
    "CommitRevealEnabledError",
    "CommitmentDecodeError",
    "CommitmentError",
    "CommitmentMismatchError",
    "ConfigError",
    "DbLayerError",
    "DuplicateModelError",
    "EmbargoedError",
    "EngineHashMismatchError",
    "EvaluationError",
    "LabelResponseError",
    "LabellingError",
    "ManifestViolationError",
    "ModelResponseError",
    "NotAuthorizedError",
    "PrometheonError",
    "RegistryError",
    "RevisionFormatError",
    "ScoringError",
    "SecurityError",
    "SetWeightsFailedError",
    "SignatureError",
    "SignatureFormatError",
    "SignatureVerificationError",
    "SnapshotNotReadyError",
    "SubtensorError",
    "WeightSubmissionError",
    "WeightsVersionMismatchError",
    "WrapperHashMismatchError",
]
