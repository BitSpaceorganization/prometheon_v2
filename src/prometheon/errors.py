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


class OutdatedConfigError(ConfigError):
    """The config was written for different scoring rules than this build runs."""

    code: str = "config.outdated"


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


class RevisionFormatError(RegistryError):
    """A revision must be a 40-character hex SHA, never a branch or tag."""

    code: str = "registry.revision_not_sha"


class ModelTooLargeError(RegistryError):
    """The weights exceed what every validator is expected to be able to run.

    A size ceiling is unusual as a *protocol* rule, and it is one here for a
    structural reason: every validator downloads and runs every eligible model,
    so an oversized model is not one miner overreaching on their own hardware
    budget. It is work the whole subnet attempts and fails at simultaneously.
    """

    code = "registry.model_too_large"


class DuplicateModelError(RegistryError):
    """Another hotkey committed the same weights first."""

    code: str = "registry.duplicate_model"


class RepoUnavailableError(RegistryError):
    """The repository or revision is gone, private or gated: the miner's to fix.

    Distinct from :class:`RegistryUnavailableError` because the two must be
    treated oppositely. A repository a miner deleted is that miner's loss on
    every validator alike; Hugging Face failing to answer one validator is that
    validator's fault and must not decide who is scored.
    """

    code: str = "registry.repo_unavailable"


class RegistryUnavailableError(RegistryError):
    """Hugging Face did not answer after the retries: a validator-side fault."""

    code: str = "registry.upstream_unavailable"


class FingerprintMismatchError(RegistryError):
    """The weights at the frozen revision are not the ones the epoch names."""

    code: str = "registry.fingerprint_mismatch"


# ---------------------------------------------------------------------------
# The hub: the signed daily epoch and the results it stores
# ---------------------------------------------------------------------------


class HubError(PrometheonError):
    """A call to the hub failed, or what it served could not be accepted.

    Carries the HTTP status and the hub's own error code when the failure came
    from a response, so a caller can tell "not built yet" from "you are not a
    validator" without parsing prose.
    """

    code: str = "hub.error"

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


class EpochNotReadyError(HubError):
    """The hub has not built this day's epoch yet. Wait and retry."""

    code: str = "hub.epoch_not_ready"


class EpochMissingError(HubError):
    """The hub says this day will never have an epoch."""

    code: str = "hub.epoch_missing"


class RecordMissingError(HubError):
    """The hub says the owner validator's record for this day will never exist."""

    code: str = "hub.record_missing"


class EpochVerificationError(HubError):
    """The epoch's signature, schema, or contents could not be accepted.

    Never retried: an epoch that fails verification is a pinned-key mismatch, a
    version skew, or tampering, and each of those needs a person.
    """

    code: str = "hub.epoch_invalid"


class FeedFileError(HubError):
    """A content file did not match what the signed epoch says it holds."""

    code: str = "hub.feed_invalid"


class HubAuthError(HubError):
    """The hub refused the request's signature, or this hotkey's permit."""

    code: str = "hub.not_authorized"


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


class LabellingError(PrometheonError):
    code: str = "labelling.error"


class LabelResponseError(LabellingError):
    """The labeller returned something that is not a usable answer.

    Raised when the answer does not echo the post's id, carries other keys, or
    its verdict is not a boolean. Never repaired by guessing: the post is asked
    once more, and dropped from the corpus if it still fails.
    """

    code: str = "labelling.response_invalid"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class EvaluationError(PrometheonError):
    code: str = "evaluation.error"


class ModelResponseError(EvaluationError):
    """A model's response did not satisfy the engine contract.

    Per the scoring rules this is not fatal: the batch is scored as incorrect
    and evaluation continues with the next one.
    """

    code: str = "evaluation.response_invalid"


class ValidatorFaultError(PrometheonError):
    """Something on this validator's side failed, so this cycle must not submit.

    A miner's model that will not download or load scores zero on every
    validator alike, which is fair. The same symptom caused by this validator,
    a missing package, a full disk, a GPU driver, an unreachable Hugging Face,
    would score every miner zero here only, and submitting that vector pays the
    burn hotkey for a validator's outage (v2 did exactly that: B10). So it stops
    the cycle instead, and the previous vector keeps being re-posted.
    """

    code: str = "validator.fault"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ScoringError(PrometheonError):
    code: str = "scoring.error"


__all__ = [
    "CanonicalEncodingError",
    "ChainError",
    "CommitRevealEnabledError",
    "CommitmentDecodeError",
    "CommitmentError",
    "ConfigError",
    "DuplicateModelError",
    "EpochMissingError",
    "EpochNotReadyError",
    "EpochVerificationError",
    "EvaluationError",
    "FeedFileError",
    "FingerprintMismatchError",
    "HubAuthError",
    "HubError",
    "LabelResponseError",
    "LabellingError",
    "ManifestViolationError",
    "ModelResponseError",
    "ModelTooLargeError",
    "OutdatedConfigError",
    "PrometheonError",
    "RecordMissingError",
    "RegistryError",
    "RegistryUnavailableError",
    "RepoUnavailableError",
    "RevisionFormatError",
    "ScoringError",
    "SecurityError",
    "SetWeightsFailedError",
    "SignatureError",
    "SignatureFormatError",
    "SignatureVerificationError",
    "SubtensorError",
    "ValidatorFaultError",
    "WeightSubmissionError",
    "WeightsVersionMismatchError",
]
