"""Copy mode: submitting the owner validator's weights instead of computing your own.

A validator that doesn't want to run the work sets ``[validator] mode = "copy"``.
Each day it fetches the owner validator's signed record from the hub's
validators-only route and submits that record's weights unchanged. It needs no
GPU and no labelling key.

Be plain about what that means. **The owner validator's results decide the
weights of every validator that copies it.** A copying validator adds stake to
the owner validator's opinion, not an opinion of its own. If the owner
validator scores a day wrongly, every copy submits the same mistake, and
consensus can't see it, because the copies agree with it by construction.

So the checks here are about *provenance*, the only thing a copy can check
mechanically, and they are strict:

- the record is signed by ``[validator] copy_from``, the owner validator's
  hotkey, and by no other: the hub cannot hand a copying validator anybody
  else's record, or a record it made itself;
- that hotkey is registered and holds a validator permit on this netuid now;
- the record is for the day, the netuid and the scoring rules being run;
- it names the same signed epoch this validator can verify for the day, so a
  stale or substituted day is refused rather than submitted;
- the burn is submitted as part of the vector, exactly as the owner sent it.
"""

from __future__ import annotations

import datetime as dt

from prometheon.chain.metagraph import MetagraphView
from prometheon.errors import ScoringError, SignatureError
from prometheon.hub.auth import verify_evaluation
from prometheon.hub.models import EvaluationSubmission
from prometheon.version import SCORING_VERSION


class CopyRefusedError(ScoringError):
    """A fetched record could not be trusted enough to submit."""

    code = "scoring.copy_refused"


def verify_copied_record(
    submission: EvaluationSubmission,
    *,
    owner: str,
    day: dt.date,
    netuid: int,
    metagraph: MetagraphView,
    burn_hotkey: str,
    epoch_content_hash: str | None = None,
) -> dict[str, int]:
    """Check the owner validator's record and return its ``{hotkey: weight}`` map.

    The map includes ``burn_hotkey`` carrying the record's ``burned_weight``.
    The burn is part of the vector, not a leftover: it is the share the owner
    validator deliberately did not pay out, and dropping it renormalises every
    miner upward by exactly the burn fraction. That produces a vector the owner
    never submitted while every provenance check still passes, which is the
    worst shape a bug here can take -- a 20% burn became a silent 1.25x on every
    miner, and the only visible symptom was the copying validator's vtrust
    sitting at 0.8.

    ``burn_hotkey`` is a parameter because the record carries the *amount* and
    not the address; the caller reads the subnet owner from chain. Passing it
    is therefore required rather than optional -- a default would let a caller
    reintroduce the same omission by forgetting it.

    Raises :class:`CopyRefusedError` on any failure. Never returns a partial
    result: a record that fails one check is not a record to submit most of.
    """
    if submission.validator_hotkey != owner:
        raise CopyRefusedError(
            f"record was published by {submission.validator_hotkey!r}, but copy mode copies "
            f"the owner validator {owner!r} named in [validator] copy_from, and no other hotkey"
        )
    if submission.date != day:
        raise CopyRefusedError(
            f"record is for {submission.date.isoformat()}, not {day.isoformat()}"
        )
    if submission.netuid != netuid:
        raise CopyRefusedError(f"record is for netuid={submission.netuid}, not netuid={netuid}")
    if submission.scoring_version != SCORING_VERSION:
        # v2 never compared this, so a record under other rules was copied
        # unnoticed (B26).
        raise CopyRefusedError(
            f"record was computed under {submission.scoring_version!r}, but this build "
            f"implements {SCORING_VERSION!r}"
        )

    try:
        verify_evaluation(submission)
    except SignatureError as exc:
        raise CopyRefusedError(
            f"record's signature does not verify against {owner!r}: {exc}. "
            "The hub cannot mint a record for a hotkey it does not hold, so "
            "this is either corruption in transit or a forgery"
        ) from exc

    if not metagraph.is_registered(owner):
        raise CopyRefusedError(
            f"the owner validator {owner!r} is not registered on netuid={netuid}; a "
            "deregistered validator's weights are not worth submitting"
        )
    # By hotkey, never by indexing the permit column with a uid: the columns
    # are in the SDK's row order, which need not be uid order (S10).
    if not metagraph.has_validator_permit(owner):
        raise CopyRefusedError(
            f"the owner validator {owner!r} holds no validator permit on netuid={netuid}"
        )

    if epoch_content_hash is not None and submission.epoch_content_hash != epoch_content_hash:
        raise CopyRefusedError(
            f"record names epoch {submission.epoch_content_hash[:12]}, but the signed epoch "
            f"for the day is {epoch_content_hash[:12]}; the two describe different days"
        )

    # Rows carry earned units only, never burn, so they sum with the burn to the
    # owner validator's vector exactly. Zero rows are left out, as it left them
    # out of what it sent.
    weights = {result.hotkey: result.weight for result in submission.results if result.weight}
    if not weights and not submission.burned_weight:
        raise CopyRefusedError("record carries no weights to submit")
    if submission.burned_weight:
        # Added, not merely acknowledged. An owner validator whose burn hotkey
        # also earned has a row of its own, and its burn adds to it.
        weights[burn_hotkey] = weights.get(burn_hotkey, 0) + submission.burned_weight
    return weights


__all__ = ["CopyRefusedError", "verify_copied_record"]
