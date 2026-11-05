"""Verifying the hub's signed epoch, and the rules read off it.

The epoch is what makes every validator score the same day: the same frozen
miners, the same frozen models, the same content. So it is checked before
anything in it is believed, in this order:

1. The signature, over the exact bytes received, against a public key pinned in
   the validator's own config. The hub cannot change a key a validator trusts
   by serving a different one.
2. The schema, so a format this client does not implement is refused rather
   than half read.
3. That it is the day, the netuid and the scoring rules asked for.

What a validator still checks for itself, past the signature, is everything it
can: content hashes and line counts, every feed line, each model's repository
and fingerprint on Hugging Face. What it cannot check is who BitFan counts as a
verified person and which hotkeys were registered at the frozen block; those it
takes from the signed epoch, and the docs say so.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from prometheon.errors import CanonicalEncodingError, EpochVerificationError, SignatureError
from prometheon.hub.models import Epoch, EpochMiner, EpochModel, Payment
from prometheon.security.canonical_json import to_canonical_bytes
from prometheon.security.signatures import verify_ed25519


def epoch_signing_bytes(raw: Mapping[str, Any]) -> bytes:
    """The bytes the hub signs: canonical JSON of the epoch minus ``signature``.

    ``key_id`` is inside them. No domain prefix: the hub's key signs nothing
    else, the same shape as BitFan's feed manifest.
    """
    return to_canonical_bytes({key: value for key, value in raw.items() if key != "signature"})


def epoch_content_hash(raw: Mapping[str, Any]) -> str:
    """sha256 of the signed bytes. Names the epoch in published records."""
    return hashlib.sha256(epoch_signing_bytes(raw)).hexdigest()


def verify_epoch(raw: Any, *, keys: Mapping[str, str]) -> Epoch:
    """Check an epoch's signature against the pinned keys, then parse it.

    ``keys`` maps ``key_id`` to a 64-hex Ed25519 public key, from config. Raises
    :class:`EpochVerificationError` on every failure, naming which check it was.
    """
    if not isinstance(raw, dict):
        raise EpochVerificationError("the epoch is not a JSON object")
    if not keys:
        raise EpochVerificationError(
            "no hub public key is pinned. Set [hub.epoch_keys] in your config to the key "
            "published with this release; an epoch cannot be trusted without one"
        )
    key_id = raw.get("key_id")
    signature = raw.get("signature")
    if not isinstance(key_id, str) or key_id not in keys:
        raise EpochVerificationError(
            f"the epoch is signed with key_id {key_id!r}, which this config does not pin "
            f"(pinned: {sorted(keys)}). If the hub rotated its key, update [hub.epoch_keys] "
            "from the release notes; never copy a key from the hub's own response"
        )
    if not isinstance(signature, str):
        raise EpochVerificationError("the epoch carries no signature")
    try:
        message = epoch_signing_bytes(raw)
    except CanonicalEncodingError as exc:
        raise EpochVerificationError(f"the epoch cannot be canonicalised: {exc}") from exc
    try:
        verify_ed25519(public_key_hex=keys[key_id], signature_hex=signature, message=message)
    except SignatureError as exc:
        raise EpochVerificationError(
            f"the epoch's signature does not verify against pinned key {key_id!r}: {exc}"
        ) from exc
    try:
        return Epoch.model_validate(raw)
    except ValidationError as exc:
        raise EpochVerificationError(
            f"the signed epoch is not one this client accepts: {exc}"
        ) from exc


def check_epoch_is_for(epoch: Epoch, *, day: dt.date, netuid: int, scoring_version: str) -> None:
    """Refuse an epoch for another day, another subnet, or other rules."""
    if epoch.day != day:
        raise EpochVerificationError(
            f"asked for the epoch of {day.isoformat()} and was served {epoch.day.isoformat()}"
        )
    if epoch.netuid != netuid:
        raise EpochVerificationError(
            f"the epoch is for netuid={epoch.netuid}, this validator runs netuid={netuid}"
        )
    if epoch.scoring_version != scoring_version:
        raise EpochVerificationError(
            f"the epoch was frozen under {epoch.scoring_version!r}, but this build implements "
            f"{scoring_version!r}. Update to the release that implements it rather than "
            "scoring a day under rules it was not frozen for"
        )


def is_eligible(miner: EpochMiner, *, min_verified_active_members: int, min_members: int) -> bool:
    """The one entry gate, for both pools.

    A miner in the epoch's list is eligible with at least
    ``min_verified_active_members`` identity-verified active members **and** at
    least ``min_members`` members in total. Both floors come from the
    validator's config and both counts from the signed epoch. The hub's own
    ``eligible`` flag is for its dashboard and is not consulted.
    """
    return (
        miner.verified_active_members >= min_verified_active_members
        and miner.members >= min_members
    )


def eligible_miners(
    epoch: Epoch, *, min_verified_active_members: int, min_members: int
) -> tuple[EpochMiner, ...]:
    return tuple(
        miner
        for miner in epoch.miners
        if is_eligible(
            miner, min_verified_active_members=min_verified_active_members, min_members=min_members
        )
    )


def paid_models(epoch: Epoch) -> tuple[EpochModel, ...]:
    """Frozen models whose fee is covered, as the hub published it."""
    return tuple(model for model in epoch.models if model.payment is Payment.PAID)


__all__ = [
    "check_epoch_is_for",
    "eligible_miners",
    "epoch_content_hash",
    "epoch_signing_bytes",
    "is_eligible",
    "paid_models",
    "verify_epoch",
]
