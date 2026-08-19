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


class DbLayerConfig(BaseModel):
    """The subnet DB layer: the only bridge to platform-sourced data."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    base_url: str = Field(min_length=1)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


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
    #: rejects an explicit ``0`` outright — and a default that cannot run with
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
        surfaces *after* the snapshot is fetched and hours into a cycle, as a
        provider error with no hint that the fix is one line of config. It is
        knowable the moment the file is read, so it is refused here.
        """
        if self.model.startswith("gpt-5") and self.temperature == 0:
            raise ValueError(
                f"model {self.model!r} rejects an explicit temperature of 0 "
                "(HTTP 400: 'Only the default (1) value is supported'). "
                "Set `temperature = 1.0` under [labelling], or choose a model "
                "that accepts 0 — 0 is what keeps ground truth from wandering "
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


class ScoringConfig(BaseModel):
    """Every constant that moves emissions.

    Defaults are the agreed reward design. Changing one changes payouts, so
    each carries the reasoning that produced it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: Share of miner emission that always burns to the subnet owner (uid 0)
    #: before any miner is paid. The remaining ``10000 - this`` is the pool the
    #: dataset and model halves divide among miners by score. At 4000, 40% burns
    #: every day and miners compete for the other 60%; the burn line on the
    #: dashboard is the subnet stating that share explicitly rather than letting
    #: the chain renormalise it away.
    miner_burn_share_bp: int = Field(default=4000, ge=0, le=10000)

    #: Dataset contribution's share of the *miner pool* (what is left after the
    #: burn share above); model performance takes the rest. At 6667 against a
    #: 6000 pool that is 40% of total emission for data and 20% for models —
    #: two to one, which is as close as basis points of the pool get to it.
    #:
    #: The weighting is deliberate. Data is what every model is measured
    #: against, it accrues daily from real users, and it is the half a miner
    #: can earn without a deployed model at all.
    dataset_share_bp: int = Field(default=6667, ge=0, le=10000)

    #: Only the top contributors are paid, proportionally to score. With fewer
    #: than this many eligible contributors the pool divides among those
    #: present rather than burning the remainder.
    dataset_top_n: int = Field(default=10, ge=1)

    #: Rank ratios within the model half of the miner pool: 30/15/5 is the top
    #: three in 6:3:1, renormalised to fill that half across the models actually
    #: present. With fewer valid models the shares renormalise across those
    #: present rather than burning the difference.
    #:
    #: Constrained here rather than at first use. An empty or negative-bearing
    #: tuple used to validate cleanly and only fail at ``allocate_model_pool``,
    #: which is the *last* step of a cycle. A typo in the TOML cost a full day
    #: of labelling and evaluation before anything complained.
    model_rank_shares_bp: tuple[int, ...] = Field(default=(3000, 1500, 500), min_length=1)

    #: The smallest number of scored items a model may hold a rank share on.
    #:
    #: Accuracy over one item is 100% or 0% and means nothing either way, but
    #: it sorts above a model that got 4999 of 5000 right, and takes twice its
    #: emission. That is reachable rather than theoretical: a miner's own test
    #: content is excluded from its corpus, so a miner who supplies almost all
    #: of a day's test content is scored on almost nothing. Below this floor a
    #: model is not scored at all, and the rank shares renormalise across the
    #: models that were actually measured.
    model_min_scored_items: int = Field(default=100, ge=1)

    #: Efficiency weight. Capped low: accuracy is what is being scored, so
    #: efficiency must never overturn an accuracy gap wider than this fraction.
    efficiency_lambda_bp: int = Field(default=1500, ge=0, le=10000)

    #: Below this spread the efficiency term is noise rather than signal, so it
    #: switches off entirely instead of randomising close rankings.
    efficiency_min_cv_bp: int = Field(default=1000, ge=0, le=10000)

    @field_validator("model_rank_shares_bp")
    @classmethod
    def _shares_are_payable(cls, shares: tuple[int, ...]) -> tuple[int, ...]:
        if any(share < 0 for share in shares):
            raise ValueError(f"model rank shares must not be negative, got {shares}")
        if sum(shares) == 0:
            raise ValueError("model rank shares are all zero; no model could ever be paid")
        return shares


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
    db: DbLayerConfig
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
    "DbLayerConfig",
    "EvaluationConfig",
    "LabellingConfig",
    "MinerRuntimeConfig",
    "ScoringConfig",
    "ValidatorRuntimeConfig",
    "WalletConfig",
    "load_config",
]
