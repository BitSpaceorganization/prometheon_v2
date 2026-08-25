"""Taking another validator's published scores instead of computing your own.

A validator may run ``score_source = "endpoint"``: fetch a signed record another
validator published for the day and submit its weights unchanged. It needs no
labelling budget and no GPU.

What that buys and what it costs are worth separating, because nothing
downstream can tell a mirrored vector from an earned one.

**It does not make the scores trustworthy.** The signature proves which hotkey
computed the vector. It says nothing about whether the vector is right. A
provider that scores badly, or dishonestly, is mirrored exactly as faithfully as
one that scores well.

**It removes this validator from consensus.** Consensus works by validators
disagreeing when one of them is wrong. A field where most weight mirrors one
provider cannot detect that provider being wrong: the agreement is manufactured
rather than earned, and vtrust -- which measures agreement -- reads a captured
subnet as a healthy one.

So the checks here are about *provenance*, and are deliberately strict, because
provenance is the only thing that can be checked mechanically:

- the record is signed by the hotkey the operator pinned, not by whoever the DB
  layer felt like serving;
- that hotkey holds a validator permit on this netuid right now;
- the record is for the day and the netuid being scored;
- it describes the same snapshot this validator fetched, so a stale or
  substituted day is refused rather than submitted.
"""

from __future__ import annotations

import datetime as dt

from prometheon.chain.metagraph import MetagraphView
from prometheon.dbclient.auth import verify_evaluation
from prometheon.dbclient.models import EvaluationSubmission
from prometheon.errors import ScoringError, SignatureError


class MirroredScoreError(ScoringError):
    """A fetched record could not be trusted enough to submit."""

    code = "scoring.mirrored_rejected"


def verify_mirrored(
    submission: EvaluationSubmission,
    *,
    provider: str,
    day: dt.date,
    netuid: int,
    metagraph: MetagraphView,
    snapshot_content_hash: str | None = None,
) -> dict[str, int]:
    """Check a published record and return its ``{hotkey: weight}`` map.

    Raises :class:`MirroredScoreError` on any failure. Never returns a partial
    result: a record that fails one check is not a record to submit most of.
    """
    if submission.validator_hotkey != provider:
        raise MirroredScoreError(
            f"record was published by {submission.validator_hotkey!r} but this "
            f"validator mirrors {provider!r}; refusing to submit somebody else's "
            "vector under a pinned provider"
        )
    if submission.date != day:
        raise MirroredScoreError(
            f"record is for {submission.date.isoformat()}, not {day.isoformat()}"
        )
    if submission.netuid != netuid:
        raise MirroredScoreError(f"record is for netuid={submission.netuid}, not netuid={netuid}")

    try:
        verify_evaluation(submission)
    except SignatureError as exc:
        raise MirroredScoreError(
            f"record's signature does not verify against {provider!r}: {exc}. "
            "The DB layer cannot mint a record for a hotkey it does not hold, so "
            "this is either corruption in transit or a forgery"
        ) from exc

    uid = metagraph.uid_for(provider)
    if uid is None:
        raise MirroredScoreError(
            f"provider {provider!r} is not registered on netuid={netuid}; a "
            "deregistered validator's scores are not worth submitting"
        )
    if not metagraph.validator_permit[uid]:
        raise MirroredScoreError(
            f"provider {provider!r} holds no validator permit on netuid={netuid}"
        )

    if (
        snapshot_content_hash is not None
        and submission.snapshot_content_hash != snapshot_content_hash
    ):
        raise MirroredScoreError(
            f"record describes snapshot {submission.snapshot_content_hash[:12]}… "
            f"but this validator fetched {snapshot_content_hash[:12]}…; the two "
            "were served different corpora for the same day"
        )

    weights = {result.hotkey: result.weight for result in submission.results}
    if submission.burned_weight:
        weights = dict(weights)
    if not weights and not submission.burned_weight:
        raise MirroredScoreError("record carries no weights to submit")
    return weights


__all__ = ["MirroredScoreError", "verify_mirrored"]
