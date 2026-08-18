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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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
    #: Sampling temperature for the labelling request. ``0`` keeps ground truth
    #: from wandering between validators, which is the default. Some models
    #: (the gpt-5 reasoning line among them) accept only their own default and
    #: reject an explicit ``0``; set this to their supported value, or to
    #: ``null`` to omit the field entirely for such a model.
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=3, ge=0)


class EvaluationConfig(BaseModel):
    """Calling miners' deployed models."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=100, ge=1, le=100)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    #: One retry on transport error, then the batch is scored incorrect. Not
    #: configurable upward: a model needing many retries is not serving.
    transport_retries: int = Field(default=1, ge=0, le=1)
    #: Environment variable holding the Chutes key a **validator** infers with.
    #:
    #: This is the subnet owner's key, not the miner's. A miner uses their own
    #: key to build and deploy their chute; a validator never sees it and must
    #: not, since a key that could redeploy the model under evaluation would let
    #: whoever holds it change what is being scored mid-cycle.
    #:
    #: One key across the field also keeps inference cost and rate limits
    #: attributable to the subnet rather than to whichever miner happened to be
    #: sampled.
    chutes_api_key_env: str = Field(default="CHUTES_API_KEY", min_length=1)


class ScoringConfig(BaseModel):
    """Every constant that moves emissions.

    Defaults are the agreed reward design. Changing one changes payouts, so
    each carries the reasoning that produced it.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: Share of miner emission that always burns to the subnet owner (uid 0)
    #: before any miner is paid. The remaining ``10000 - this`` is the pool the
    #: dataset and model halves divide among miners by score. At 6000, 60% burns
    #: every day and miners compete for the other 40%; the burn line on the
    #: dashboard is the subnet stating that share explicitly rather than letting
    #: the chain renormalise it away.
    miner_burn_share_bp: int = Field(default=6000, ge=0, le=10000)

    #: Dataset contribution takes half of the *miner pool* (what is left after
    #: the burn share above), model performance the other half.
    dataset_share_bp: int = Field(default=5000, ge=0, le=10000)

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

    chutes_user: str = Field(default="", max_length=128)
    chute_id: str = Field(default="", max_length=128)
    hf_repo: str = Field(default="", max_length=256)


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
