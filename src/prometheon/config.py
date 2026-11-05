"""TOML configuration for miners and validators.

Unknown keys are rejected rather than ignored. A typo in a config file should
fail at load with the offending field named, not silently take a default and
produce a subtly different weight vector three hours later.

Every reward-affecting constant lives here with its default, so the values that
decide emissions are visible in one place rather than scattered through the
scoring modules.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from prometheon.errors import ConfigError
from prometheon.security.signatures import is_ed25519_public_key

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib


class ChainNetwork(str, Enum):
    LOCAL = "local"
    TEST = "test"
    FINNEY = "finney"


class ChainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    network: ChainNetwork
    netuid: int = Field(ge=0)
    version_key: int = Field(default=0, ge=0)
    #: Submitting under a stale version key produces weights the chain ignores.
    #: Failing closed is right everywhere; it is configurable only so a
    #: localnet can run without matching a deployed key.
    fail_on_weights_version_mismatch: bool = True
    #: Waive the gate that requires the SDK to accept a ``mechid``. Localnet
    #: only. On a live network the weights land on whatever mechanism the chain
    #: defaults to, and nothing says so, because the extrinsic succeeds either
    #: way. This defaults to failing closed and should stay that way anywhere
    #: real.
    allow_missing_mechid: bool = False


class WalletConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    hotkey: str = Field(min_length=1)


class HubConfig(BaseModel):
    """The hub: where the signed daily epoch comes from and results go."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    base_url: str = Field(min_length=1)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    #: The hub's Ed25519 public keys, by ``key_id``: 64 lowercase hex each.
    #:
    #: Pinned here, from the release notes, and never taken from the hub itself:
    #: a key the hub could change is a signature that proves nothing. More than
    #: one entry lets a key rotation overlap. Empty is allowed so a miner's config
    #: still loads; a validator refuses to score until a key is pinned.
    epoch_keys: dict[str, str] = Field(default_factory=dict)

    @field_validator("epoch_keys")
    @classmethod
    def _keys_are_ed25519(cls, keys: dict[str, str]) -> dict[str, str]:
        for key_id, public_key in keys.items():
            if not key_id or len(key_id) > 64:
                raise ValueError(f"epoch key id {key_id!r} must be 1 to 64 characters")
            if not is_ed25519_public_key(public_key):
                raise ValueError(
                    f"epoch key {key_id!r} must be 64 lowercase hex characters (a raw "
                    "32-byte Ed25519 public key)"
                )
        return keys


class LabellingConfig(BaseModel):
    """Ground-truth labelling."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(default="gpt-5.6-sol", min_length=1)
    base_url: str = Field(default="https://api.openai.com/v1", min_length=1)
    api_key_env: str = Field(default="OPENAI_API_KEY", min_length=1)
    batch_size: int = Field(default=100, ge=1, le=100)
    #: Sampling temperature for the labelling request.
    #:
    #: ``0`` is what keeps ground truth from wandering between validators, and
    #: is the right value for any model that accepts it. It is *not* the default
    #: here only because the default ``model`` is a gpt-5 reasoning model, which
    #: rejects an explicit ``0`` outright, and a default that cannot run with
    #: the neighbouring default is a trap, not a default. Set ``0`` explicitly
    #: when you point ``model`` at something that allows it.
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _temperature_suits_the_model(self) -> LabellingConfig:
        """Refuse a model/temperature pair the endpoint will reject.

        The gpt-5 reasoning line accepts only its own default of 1 and answers
        an explicit 0 with HTTP 400. Left to be discovered at run time that
        surfaces *after* the epoch is fetched and hours into a cycle, as a
        provider error with no hint that the fix is one line of config. It is
        knowable the moment the file is read, so it is refused here.
        """
        if self.model.startswith("gpt-5") and self.temperature == 0:
            raise ValueError(
                f"model {self.model!r} rejects an explicit temperature of 0 "
                "(HTTP 400: 'Only the default (1) value is supported'). "
                "Set `temperature = 1.0` under [labelling], or choose a model "
                "that accepts 0, 0 is what keeps ground truth from wandering "
                "between validators, so prefer it where the model allows it."
            )
        return self


class EvaluationConfig(BaseModel):
    """Running miners' models on this validator's own hardware."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=100, ge=1, le=100)

    #: Ceiling on a submitted model's weights, in bytes.
    #:
    #: Every validator downloads and runs every eligible model, so this is not
    #: a miner's private trade-off: it decides what hardware validating the
    #: subnet requires. 24 GiB leaves headroom for KV cache and activations on
    #: a 32 GB card, which is the floor a validator is expected to own.
    #:
    #: Enforced from the Hugging Face API before anything is downloaded, so an
    #: oversized model costs the subnet one request rather than an hour of
    #: bandwidth on every validator at once.
    max_weight_bytes: int = Field(default=24 * 1024**3, gt=0)

    #: Torch dtype the weights are loaded in, pinned rather than inferred.
    #:
    #: Two validators must reach the same verdict on the same item, and dtype
    #: is the largest avoidable source of divergence: the same checkpoint in
    #: bfloat16 and float16 puts different probability mass on YES and NO, so
    #: near-ties can flip between validators that only differ in a default.
    #: Residual disagreement from differing GPU kernels is left to consensus,
    #: which is what consensus is for; a silent dtype difference is not.
    torch_dtype: str = Field(default="float16", min_length=1)

    #: Where to run. ``auto`` picks CUDA when present and falls back to CPU,
    #: which is unusably slow for a real cycle but keeps a dry run possible on
    #: a machine without a GPU.
    device: str = Field(default="auto", min_length=1)

    #: Seconds a single model's evaluation may take before it is abandoned and
    #: scored on whatever it completed. A model that cannot finish the corpus
    #: is not a model the subnet can serve.
    model_timeout_seconds: float = Field(default=3600.0, gt=0)

    #: Delete each model's downloaded snapshot once it has been scored.
    #:
    #: Off by default: the cache is keyed by revision, so keeping it means a
    #: miner who re-commits an unchanged model costs one HEAD request and a
    #: validator restarting mid-cycle does not re-download the day.
    #:
    #: Turn it on when the disk is smaller than the field. One day's eligible
    #: set is routinely over 100 GiB of weights, and nothing evicts from the
    #: cache, so a host with less free space than that cannot finish a cycle;
    #: the failure arrives as a disk-full error partway through scoring. With
    #: this set, the requirement is the largest single model rather than the sum
    #: of all of them, paid for by re-downloading each model every day.
    discard_checkpoints: bool = Field(default=False)


class ScoreSource(str, Enum):
    """Where a validator's weight vector comes from."""

    #: Score the day yourself: label the corpus, run every eligible model, and
    #: reduce it here. Everything the subnet's trust model assumes.
    LOCAL = "local"

    #: Take a published, signed record from another validator and submit its
    #: weights. Costs no labelling and no GPU, and makes this validator's
    #: opinion a copy of somebody else's.
    ENDPOINT = "endpoint"


class ScoringConfig(BaseModel):
    """Every constant that moves emissions.

    Defaults are the agreed reward design. Changing one changes payouts, so
    each carries the reasoning that produced it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: The data contribution pool, as a share of **total** emission: 70%.
    #:
    #: It starts high because inviting real people and getting them to write
    #: test content costs far more today than training a text model. As user
    #: numbers grow and model requirements get harder, the share moving to model
    #: training is expected to reach 80 to 90%.
    #:
    #: Every validator must agree on this number, as on everything in this
    #: section. It is not enforced by consensus, so one running a different
    #: value computes a different vector from identical inputs and loses vtrust
    #: for a configuration difference.
    dataset_share_bp: int = Field(default=7000, ge=0, le=10000)

    #: Only the top contributors are paid, proportionally to score. With fewer
    #: than this many scoring above zero, the pool divides among those present.
    dataset_top_n: int = Field(default=10, ge=1)

    #: The model pool: a fixed share of **total** emission per rank, 25%, 4% and
    #: 1% for the first three models. Shares, not ratios. A rank with no model to
    #: fill it is not handed to the ranks above it; it burns. So one weak model
    #: alone collects 25% and the other 5% burns.
    #:
    #: Constrained here rather than at first use. An empty or negative-bearing
    #: tuple used to validate cleanly and only fail at ``allocate_model_pool``,
    #: which is the *last* step of a cycle. A typo in the TOML cost a full day
    #: of labelling and evaluation before anything complained.
    model_rank_shares_bp: tuple[int, ...] = Field(default=(2500, 400, 100), min_length=1)

    #: The entry gate, for both pools: a miner counts only with at least this
    #: many identity-verified active members in its group. The hub publishes the
    #: count and its own flag, and both must clear; see
    #: :func:`prometheon.hub.epoch.is_eligible`.
    min_verified_active_members: int = Field(default=10, ge=0)

    #: Two items whose normalised text has 64-bit SimHashes within this many bits
    #: are one item, and only the earliest counts. Bounded, because each bit
    #: allowed narrows the band index that makes the comparison fast.
    near_duplicate_max_hamming: int = Field(default=3, ge=0, le=7)

    #: The smallest number of scored items a model may hold a rank share on.
    #:
    #: Accuracy over one item is 100% or 0% and means nothing either way, but
    #: it sorts above a model that got 4999 of 5000 right, and takes twice its
    #: emission. That is reachable rather than theoretical: a miner's own test
    #: content is excluded from its corpus, so a miner who supplies almost all
    #: of a day's test content is scored on almost nothing. Below this floor a
    #: model is not ranked at all.
    model_min_scored_items: int = Field(default=100, ge=1)

    #: Efficiency weight. Capped low: accuracy is what is being scored, so
    #: efficiency must never overturn an accuracy gap wider than this fraction.
    efficiency_lambda_bp: int = Field(default=1500, ge=0, le=10000)

    #: Below this spread the efficiency term is noise rather than signal, so it
    #: switches off entirely instead of randomising close rankings.
    efficiency_min_cv_bp: int = Field(default=1000, ge=0, le=10000)

    #: Where the weight vector comes from. ``local`` computes it here.
    #:
    #: ``endpoint`` fetches a signed record published by ``score_provider``
    #: and submits its weights unchanged. That is a real option -- it needs no
    #: labelling budget and no GPU -- and it is worth being blunt about what it
    #: costs the subnet, because nothing downstream can tell the difference.
    #:
    #: A mirroring validator contributes no independent opinion. Consensus works
    #: by validators disagreeing when one of them is wrong; a field where most
    #: weight mirrors one provider cannot detect that provider being wrong,
    #: because the agreement is manufactured rather than earned. The provider's
    #: signature proves *who* computed the vector, never that the vector is
    #: right.
    #:
    #: What it does still give you: the record is signed by the provider's
    #: hotkey and verified here, so the hub cannot forge one, and its epoch hash
    #: is checked against the signed epoch for the day, so a stale or
    #: substituted record is refused rather than submitted.
    score_source: ScoreSource = ScoreSource.LOCAL

    #: The validator whose published record to mirror when ``score_source`` is
    #: ``endpoint``. Required in that mode and ignored otherwise.
    #:
    #: Pinned by hotkey, not left to whoever answers: the record is verified
    #: against *this* address, so a hub serving somebody else's record -- or
    #: its own -- fails the signature check instead of being submitted.
    score_provider: str = ""

    @model_validator(mode="after")
    def _endpoint_mode_names_a_provider(self) -> ScoringConfig:
        """A mirroring validator must say whose record it mirrors.

        Refused at load rather than at submission: the alternative is a cycle
        that runs to the end and then has nothing to send.
        """
        if self.score_source is ScoreSource.ENDPOINT and not self.score_provider.strip():
            raise ValueError(
                "score_source = 'endpoint' requires score_provider to name the "
                "validator hotkey whose published record to mirror.\n"
                "\n"
                "A validator's record for a day is public on the hub:\n"
                "    GET <hub.base_url>/api/v1/results/<YYYY-MM-DD>/<validator hotkey>\n"
                "Read a few before you pin one.\n"
                "\n"
                "It is not filled in for you on purpose. Mirroring submits "
                "whatever that hotkey published, so choosing it is a trust "
                "decision -- and if this service picked, the service would be "
                "choosing whom you trust."
            )
        return self

    @field_validator("model_rank_shares_bp")
    @classmethod
    def _shares_are_payable(cls, shares: tuple[int, ...]) -> tuple[int, ...]:
        if any(share < 0 for share in shares):
            raise ValueError(f"model rank shares must not be negative, got {shares}")
        if sum(shares) == 0:
            raise ValueError("model rank shares are all zero; no model could ever be paid")
        return shares

    @model_validator(mode="after")
    def _shares_fit_in_the_total(self) -> ScoringConfig:
        """The shares are slices of one total, so together they cannot pass 100%.

        Less than 100% is allowed and means a standing burn: whatever no share
        names goes to the burn hotkey every cycle.
        """
        allocated = self.dataset_share_bp + sum(self.model_rank_shares_bp)
        if allocated > 10000:
            raise ValueError(
                f"dataset_share_bp + model_rank_shares_bp = {allocated} basis points, "
                "over 100% of emission. They are shares of the total, not ratios"
            )
        return self


class ValidatorRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    state_directory: Path = Field(default=Path(".validator-state"))
    submit_weights: bool = True
    dry_run: bool = False
    #: Publishing results is what makes a cycle auditable by anyone else.
    submit_results: bool = True


class MinerRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    hf_repo: str = Field(default="", max_length=256)
    #: The 40-character commit SHA to serve. Settable here because
    #: `model commit` tells a miner to "set them under [miner] in the config
    #: file" when a value is missing, and this was the one value that advice
    #: did not work for: it had no field, so following the error was impossible.
    hf_revision: str = Field(default="", max_length=40)


class Config(BaseModel):
    """The whole file."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    chain: ChainConfig
    wallet: WalletConfig
    hub: HubConfig
    labelling: LabellingConfig = LabellingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    scoring: ScoringConfig = ScoringConfig()
    validator: ValidatorRuntimeConfig = ValidatorRuntimeConfig()
    miner: MinerRuntimeConfig = MinerRuntimeConfig()


def load_config(path: Path | str) -> Config:
    """Read and validate a TOML config, or raise :class:`ConfigError`."""
    path = Path(path)
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"config could not be read: {exc}") from exc

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc


__all__ = [
    "ChainConfig",
    "ChainNetwork",
    "Config",
    "EvaluationConfig",
    "HubConfig",
    "LabellingConfig",
    "MinerRuntimeConfig",
    "ScoreSource",
    "ScoringConfig",
    "ValidatorRuntimeConfig",
    "WalletConfig",
    "load_config",
]
