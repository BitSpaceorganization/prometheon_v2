"""TOML configuration for miners and validators.

Only what is the operator's to choose: where the chain and the hub are, which
wallet signs, which labeller to pay, which card to run on, and how the
validator runs. The rules every validator applies are code, in
:mod:`prometheon.rules`, so a release carries its rules with it and an update
never needs an operator to edit this file.

Unknown keys are rejected rather than ignored. A typo in a config file should
fail at load with the offending field named, not silently take a default. The
one exception is a setting an earlier release read and this one doesn't
(:data:`RETIRED_SETTINGS`): it is ignored, with a note.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from prometheon.errors import ConfigError, OutdatedConfigError
from prometheon.security.signatures import is_ed25519_public_key
from prometheon.version import SUBMISSION_FEE_ALPHA

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib


class ChainNetwork(str, Enum):
    LOCAL = "local"
    TEST = "test"
    FINNEY = "finney"


#: A chain endpoint given as a URL rather than a network name.
_ENDPOINT_RE: Final[re.Pattern[str]] = re.compile(r"^wss?://[A-Za-z0-9.\-]+(:[0-9]{1,5})?(/.*)?$")


class ChainConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: ``finney``, ``test``, ``local``, or a ``ws://``/``wss://`` endpoint such as
    #: a local development chain. Required: there is no default network.
    network: str
    #: Required, with no default. The subnet has not been registered yet, so
    #: the shipped examples carry ``0``, which is refused here with a message
    #: rather than silently pointing a validator at the root network.
    netuid: int
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

    @field_validator("network")
    @classmethod
    def _known_network(cls, value: str) -> str:
        names = {network.value for network in ChainNetwork}
        if value in names or _ENDPOINT_RE.match(value):
            return value
        raise ValueError(
            f"network must be one of {sorted(names)} or a ws:// or wss:// endpoint, got {value!r}"
        )

    @field_validator("netuid")
    @classmethod
    def _a_real_subnet(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(
                f"netuid is {value}. Set [chain] netuid to the Prometheon netuid announced with "
                "the release; 0 is the root network, and the examples ship 0 so a copied file "
                "cannot point anywhere until it is filled in"
            )
        return value


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
    #: Labelling requests open at once. Each request carries one post, so this
    #: is what decides how fast a day is labelled. Raise it if your provider's
    #: rate limit allows; a rate-limited request is retried with backoff.
    max_in_flight: int = Field(default=8, ge=1, le=64)
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
        # Matched on the model's own name, so a provider prefix such as
        # OpenRouter's `openai/` does not hide it (B30).
        if self.model.rsplit("/", 1)[-1].startswith("gpt-5") and self.temperature == 0:
            raise ValueError(
                f"model {self.model!r} rejects an explicit temperature of 0 "
                "(HTTP 400: 'Only the default (1) value is supported'). "
                "Set `temperature = 1.0` under [labelling], or choose a model "
                "that accepts 0, 0 is what keeps ground truth from wandering "
                "between validators, so prefer it where the model allows it."
            )
        return self


class EvaluationConfig(BaseModel):
    """Running miners' models on this validator's own hardware.

    Only what is the operator's to choose: the card and how the disk is used.
    The rules a model is evaluated under, the time limit, the batch size and
    the size ceilings, are code (:mod:`prometheon.rules`).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: Torch dtype the weights are loaded in, pinned rather than inferred.
    #:
    #: Two validators must reach the same verdict on the same item, and dtype
    #: is the largest avoidable source of divergence: the same checkpoint in
    #: bfloat16 and float16 puts different probability mass on YES and NO, so
    #: near-ties can flip between validators that only differ in a default.
    #: Keep the default on a GPU. It is a setting only because a CPU, as on a
    #: local chain, needs float32.
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "float16"

    #: Where to run. ``auto`` picks CUDA when present and falls back to CPU,
    #: which is unusably slow for a real cycle but keeps a dry run possible on
    #: a machine without a GPU.
    device: str = Field(default="auto", pattern=r"^(auto|cpu|mps|cuda(:[0-9]+)?)$")

    #: Delete each model's downloaded snapshot once it has been scored.
    #:
    #: Off by default: the cache is keyed by revision, so keeping it means a
    #: miner who re-commits an unchanged model costs one HEAD request and a
    #: validator restarting mid-cycle does not re-download the day.
    #:
    #: Turn it on when the disk is smaller than the field. Nothing evicts from
    #: the cache, so a host with less free space than the day's models fails
    #: partway through scoring with a disk-full error. With this set, the
    #: requirement is the largest single model rather than the sum of all of
    #: them, paid for by re-downloading each model every day.
    discard_checkpoints: bool = Field(default=False)


class ValidatorMode(str, Enum):
    """What a validator does each day."""

    #: Label the day, run every eligible model, score, and submit the result.
    SCORE = "score"

    #: Submit the owner validator's weights unchanged. No GPU and no labelling
    #: key, and no opinion of its own: the owner validator's results decide
    #: this validator's weights. See :mod:`prometheon.scoring.copied`.
    COPY = "copy"


#: An SS58 address, as the hub and the chain write one.
_SS58_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")


class ValidatorRuntimeConfig(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    state_directory: Path = Field(default=Path(".validator-state"))
    submit_weights: bool = True
    dry_run: bool = False
    #: Publishing results is what makes a cycle auditable by anyone else.
    submit_results: bool = True
    #: ``score`` to do the day's work, ``copy`` to submit the owner validator's
    #: weights unchanged.
    mode: ValidatorMode = ValidatorMode.SCORE
    #: The owner validator's hotkey, which copy mode copies and no other. It is
    #: published with the launch; the shipped examples carry a placeholder,
    #: refused in copy mode, so a copied file cannot copy anybody by accident.
    #: Ignored in score mode.
    copy_from: str = Field(default="", max_length=64)

    @model_validator(mode="after")
    def _copy_mode_names_the_owner_validator(self) -> ValidatorRuntimeConfig:
        """Refused at load rather than at the first copy, which runs on a timer."""
        if self.mode is ValidatorMode.COPY and not _SS58_RE.match(self.copy_from.strip()):
            raise ValueError(
                "mode = 'copy' needs [validator] copy_from set to the owner validator's "
                f"hotkey, published with the launch; got {self.copy_from!r}. Copy mode "
                "submits that validator's weights unchanged, and copies no other hotkey"
            )
        return self


class CreditsConfig(BaseModel):
    """Submission credits: who is paid.

    What a credit costs is not here: it is the published rule
    :data:`~prometheon.version.SUBMISSION_FEE_ALPHA`, which the hub prices by.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: The treasury coldkey that fees are paid to. There is no built-in
    #: address: the team publishes it with the release that launches the
    #: subnet, and ``credits buy`` refuses to send anything until it is set,
    #: here or in ``PROMETHEON_TREASURY_COLDKEY``.
    treasury_coldkey: str = ""


class UpdateConfig(BaseModel):
    """Updating to new releases on its own, which is off unless turned on."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    #: Move to each new release tag on its own: ``validator schedule`` checks
    #: every 30 minutes, and so does the shipped systemd timer. Off by default,
    #: because an operator should choose to let code change under them.
    auto: bool = False
    #: systemd units to restart once a new release is current, for a process
    #: that keeps running, such as ``validator schedule`` under its own unit.
    #: The shipped units start a fresh process from ``<root>/current`` every
    #: time, so they need none.
    restart_units: tuple[str, ...] = ()


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
    validator: ValidatorRuntimeConfig = ValidatorRuntimeConfig()
    miner: MinerRuntimeConfig = MinerRuntimeConfig()
    credits: CreditsConfig = CreditsConfig()
    update: UpdateConfig = UpdateConfig()


#: Settings earlier releases read and this one doesn't, by section; ``None``
#: retires a whole section. A file that still has them loads: each is ignored,
#: with a note saying so, so an update never needs an operator to edit their
#: file. A release that retires a setting adds it here.
RETIRED_SETTINGS: Final[Mapping[str, frozenset[str] | None]] = MappingProxyType(
    {
        # Every rule moved into the code, in prometheon.rules.
        "scoring": None,
        "labelling": frozenset({"batch_size"}),
        "evaluation": frozenset(
            {"batch_size", "max_weight_bytes", "max_non_weight_bytes", "model_timeout_seconds"}
        ),
    }
)


def load_config(path: Path | str, *, notes: list[str] | None = None) -> Config:
    """Read and validate a TOML config, or raise :class:`ConfigError`.

    Settings an earlier release read and this one doesn't are dropped, and a
    line saying so is appended to ``notes``.
    """
    path = Path(path)
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"config could not be read: {exc}") from exc

    _refuse_v2_and_the_fee(raw, path)
    dropped = _drop_retired(raw)
    if dropped and notes is not None:
        notes.append(
            f"{path}: ignoring settings this release no longer reads ({', '.join(dropped)}). "
            "The rules they set are part of the release now; the lines can be deleted"
        )
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}: {exc}") from exc


def _drop_retired(raw: dict[str, Any]) -> list[str]:
    """Remove every retired setting from ``raw``; return what was removed."""
    dropped: list[str] = []
    for section, keys in RETIRED_SETTINGS.items():
        table = raw.get(section)
        if not isinstance(table, dict):
            continue
        if keys is None:
            del raw[section]
            dropped.append(f"[{section}]")
            continue
        for key in sorted(keys & table.keys()):
            del table[key]
            dropped.append(f"[{section}] {key}")
    return dropped


def _refuse_v2_and_the_fee(raw: dict[str, Any], path: Path) -> None:
    """Refuse the two kinds of file an update cannot carry forward.

    A v2 file, recognisable by its ``[db]`` section, describes a different
    system: the hub replaced the database it points at. A file that sets the
    submission fee expects it to change what ``credits buy`` sends, and it no
    longer can, so dropping it quietly would send a miner's money on a
    misunderstanding.
    """
    if "db" in raw:
        raise OutdatedConfigError(
            f"{path} was written for v2 (it has a [db] section). Copy "
            "configs/finney.example.toml again, fill in [wallet] and [chain] netuid, and read "
            "docs/validator.md"
        )
    credits = raw.get("credits")
    if isinstance(credits, dict) and "submission_fee_alpha" in credits:
        raise OutdatedConfigError(
            f"{path} sets [credits] submission_fee_alpha, which is no longer a setting. The fee "
            f"is fixed at {SUBMISSION_FEE_ALPHA} alpha per credit by the published rules, "
            "because that is what the hub credits. Delete the line"
        )


__all__ = [
    "RETIRED_SETTINGS",
    "ChainConfig",
    "ChainNetwork",
    "Config",
    "CreditsConfig",
    "EvaluationConfig",
    "HubConfig",
    "LabellingConfig",
    "MinerRuntimeConfig",
    "UpdateConfig",
    "ValidatorMode",
    "ValidatorRuntimeConfig",
    "WalletConfig",
    "load_config",
]
