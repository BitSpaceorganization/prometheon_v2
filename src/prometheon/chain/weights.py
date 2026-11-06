"""Weight submission: the policy gates, and integer units → the u16 chain vector.

Two responsibilities, both free of the SDK so they can be reasoned about and
tested on their own:

**The gates.** Three chain-side conditions make a plain ``set_weights`` either
impossible or silently wrong. Each fails closed with its own error code,
because the remediation differs: commit-reveal needs a runtime that implements
it, a version-key mismatch needs a config change, a missing ``mechid`` needs an
SDK upgrade. Failing closed is the only safe answer here, because a submission
the chain accepts and then ignores looks exactly like a successful one in the
logs.

**The conversion.** Scoring emits integers summing to whatever pool it chose;
the chain wants u16 values summing to 65535. Largest-remainder allocation makes
the output sum exactly, and a total tie-break order makes it *identical* across
validators. Two validators that computed the same units must submit the same
vector, or they disagree on chain over a rounding crumb.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from prometheon.chain.metagraph import MetagraphView
from prometheon.errors import (
    CommitRevealEnabledError,
    DuplicateWeightTargetError,
    MechidMissingError,
    WeightSubmissionError,
    WeightsVersionMismatchError,
)
from prometheon.scoring.allocation import AllocationTarget, allocate

#: What u16 weights sum to on chain.
CHAIN_WEIGHT_UNITS: Final[int] = 65_535


@dataclass(frozen=True)
class WeightAllocation:
    """One target's share, in scoring's own integer units.

    The hotkey travels alongside the UID all the way to submission. It is the
    identity the tie-break sorts on, and it is what an operator reads in the
    log when a vector looks wrong.
    """

    hotkey: str
    uid: int
    units: int


@dataclass(frozen=True)
class ChainWeightVector:
    """A resolved u16 vector, ready for the SDK.

    ``hotkeys`` mirrors ``uids`` and ``weights`` 1:1 so a caller can log
    human-readable targets without re-resolving against the metagraph.
    """

    uids: tuple[int, ...]
    weights: tuple[int, ...]
    hotkeys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.uids) == len(self.weights) == len(self.hotkeys)):
            raise WeightSubmissionError(
                f"weight vector column lengths disagree: uids={len(self.uids)}, "
                f"weights={len(self.weights)}, hotkeys={len(self.hotkeys)}"
            )

        # A duplicate UID is a lost payment. The chain stores one entry per
        # UID, so the second occurrence overwrites the first and one of the two
        # targets receives a weight nobody computed. The vector still sums to
        # 65535, so no downstream total looks wrong.
        if len(set(self.uids)) != len(self.uids):
            duplicates = sorted({uid for uid in self.uids if self.uids.count(uid) > 1})
            raise DuplicateWeightTargetError(
                f"weight vector names uid(s) {duplicates} more than once; the "
                "chain keeps one entry per uid, so the others would be silently "
                "discarded"
            )
        if len(set(self.hotkeys)) != len(self.hotkeys):
            duplicates_hk = sorted({hk for hk in self.hotkeys if self.hotkeys.count(hk) > 1})
            raise DuplicateWeightTargetError(
                f"weight vector names hotkey(s) {duplicates_hk} more than once"
            )

    def total(self) -> int:
        return sum(self.weights)


@dataclass(frozen=True)
class ChainHyperparameters:
    """The subnet hyperparameters the gates read, as of a fresh chain read."""

    weights_version: int
    commit_reveal_enabled: bool
    #: The fewest weights a validator may set (capped at the subnet's size).
    min_allowed_weights: int = 1
    #: The largest share one uid may hold, as a fraction of 65535. 65535 is no cap.
    max_weight_limit: int = 65_535


@dataclass(frozen=True)
class ChainCapabilities:
    """What the *installed* SDK can do, detected once at startup."""

    sdk_version: str
    supports_mechid: bool


def assert_submission_allowed(
    *,
    hyperparameters: ChainHyperparameters,
    capabilities: ChainCapabilities,
    configured_version_key: int,
    fail_on_weights_version_mismatch: bool = True,
    allow_missing_mechid: bool = False,
) -> None:
    """Run every pre-submission policy check, or raise the matching error.

    Ordered hardest-stop first: commit-reveal cannot be worked around at all,
    a missing ``mechid`` can be waived on a localnet, and a version-key
    mismatch is waivable by explicit config.
    """
    if hyperparameters.commit_reveal_enabled:
        # The gate exists because the SDK reroutes silently. Older SDKs turn
        # set_weights into a commit that nothing ever reveals; newer ones build
        # a timelocked commit the chain reveals rounds later. Either way the
        # weights this cycle computed do not take effect when this cycle
        # thinks they did, and the submission still reads as a success.
        raise CommitRevealEnabledError(
            "the subnet has commit-reveal weights enabled; this runtime submits "
            "plain set_weights only. The SDK reroutes the call rather than "
            "refusing it, so the weights would land on a schedule this runtime "
            "does not model"
        )

    if not capabilities.supports_mechid and not allow_missing_mechid:
        raise MechidMissingError(
            f"the installed bittensor SDK {capabilities.sdk_version!r} does not "
            "accept a mechid on set_weights. Upgrade the SDK, or set "
            "[chain] allow_missing_mechid = true; which is for localnet only, "
            "because on a live network the weights land on whatever mechanism "
            "the chain defaults to and nothing reports it"
        )

    if (
        hyperparameters.weights_version != configured_version_key
        and fail_on_weights_version_mismatch
    ):
        raise WeightsVersionMismatchError(
            f"the chain expects weights_version={hyperparameters.weights_version} "
            f"but this validator is configured with version_key="
            f"{configured_version_key}; weights submitted under a stale key are "
            "accepted and then ignored"
        )


def resolve_allocations(
    units_by_hotkey: Mapping[str, int], *, metagraph: MetagraphView
) -> list[WeightAllocation]:
    """Resolve scored hotkeys to UIDs against ``metagraph``, ordered by UID.

    Strict: an unregistered hotkey raises rather than being dropped, because
    dropping one silently redistributes its emission to everybody else.
    A caller that expects churn filters first with
    :meth:`~prometheon.chain.metagraph.MetagraphView.partition_by_registration`
    and logs what it dropped.

    Sorting by UID rather than by score makes the vector byte-identical across
    validators that agree on the numbers.
    """
    allocations: list[WeightAllocation] = []
    for hotkey, units in units_by_hotkey.items():
        if units < 0:
            raise WeightSubmissionError(
                f"hotkey {hotkey!r} was allocated {units} units; weight units are non-negative"
            )
        uid = metagraph.uid_for(hotkey)
        if uid is None:
            raise WeightSubmissionError(
                f"hotkey {hotkey!r} holds an allocation but is not registered on "
                f"netuid={metagraph.netuid} at block {metagraph.block}"
            )
        allocations.append(WeightAllocation(hotkey=hotkey, uid=uid, units=units))

    allocations.sort(key=lambda allocation: allocation.uid)
    return allocations


def to_u16_chain_vector(allocations: Sequence[WeightAllocation]) -> ChainWeightVector:
    """Scale integer units down to u16 values summing to exactly 65535.

    Largest-remainder allocation, with a total ordering on the tie-break:
    largest remainder, then largest units, then hotkey, then UID. Every element
    of that order is derived from data both validators already agree on, so the
    leftover crumbs land on the same targets everywhere.

    Raises :class:`WeightSubmissionError` for an empty input or a zero total:
    those are "there is nothing to submit", which is a decision for the runtime
    (burn, or skip the cycle), not something to paper over with a uniform
    vector nobody asked for.
    """
    if not allocations:
        raise WeightSubmissionError(
            "cannot build a weight vector from an empty allocation; the runtime "
            "must route a no-target cycle to the burn hotkey instead"
        )

    total_units = 0
    for allocation in allocations:
        if allocation.units < 0:
            raise WeightSubmissionError(
                f"hotkey {allocation.hotkey!r} has negative units {allocation.units}"
            )
        total_units += allocation.units

    if total_units == 0:
        raise WeightSubmissionError("every allocation is zero units; there is nothing to submit")

    base: list[int] = []
    remainders: list[int] = []
    for allocation in allocations:
        product = CHAIN_WEIGHT_UNITS * allocation.units
        base.append(product // total_units)
        remainders.append(product % total_units)

    leftover = CHAIN_WEIGHT_UNITS - sum(base)
    if not 0 <= leftover <= len(allocations):
        raise WeightSubmissionError(
            f"largest-remainder leftover {leftover} is outside [0, {len(allocations)}]; "
            "the allocation invariant is broken"
        )

    if leftover:
        order = sorted(
            range(len(allocations)),
            key=lambda i: (
                -remainders[i],
                -allocations[i].units,
                allocations[i].hotkey,
                allocations[i].uid,
            ),
        )
        for index in order[:leftover]:
            # A zero-unit allocation has a zero remainder and sorts behind every
            # positive one, and there are always more positive remainders than
            # leftover units, so it can never collect a crumb it did not earn.
            base[index] += 1

    return ChainWeightVector(
        uids=tuple(allocation.uid for allocation in allocations),
        weights=tuple(base),
        hotkeys=tuple(allocation.hotkey for allocation in allocations),
    )


@dataclass(frozen=True)
class ShapedAllocation:
    """A hotkey->units vector the chain will accept, and what had to change."""

    units: dict[str, int]
    #: Hotkeys given the smallest possible weight only to meet the chain's
    #: minimum number of weights, in the order they were added.
    padded: tuple[str, ...]
    #: Hotkeys whose share was cut to the chain's largest allowed share.
    capped: tuple[str, ...]


def shape_for_chain(
    units: Mapping[str, int],
    *,
    hyperparameters: ChainHyperparameters,
    metagraph: MetagraphView,
    validator_hotkey: str,
    burn_hotkey: str,
) -> ShapedAllocation:
    """Bend a vector to the subnet's weight limits, so the chain accepts it (S5).

    The chain refuses a weight vector that names fewer uids than
    ``min(subnet size, min_allowed_weights)``, or in which one uid holds a larger
    share than ``max_weight_limit / 65535``, unless it is a single self-weight.
    v2 read neither, so a day where everything burned, or a field of one,
    could be refused at the very end of a cycle. On the default limits (1 and
    65535) nothing changes.

    - **Too few uids**: the lowest-numbered registered uids not already in the
      vector, other than the validator's own, are added with the smallest
      amount that survives as a non-zero ``u16``. The units come out of the
      largest entry.
    - **A share over the cap**: it is cut to the cap. The excess is emission
      nobody may be paid, so it burns: it goes to the burn hotkey, up to the
      same cap. Only what the burn cannot hold, which happens when the burn is
      itself the entry over the cap, goes to the other entries in proportion
      to what they already hold. Repeated until nothing is over. A vector that
      cannot fit at all, because it has too few entries for the cap, raises:
      paying a large share to uids nobody scored is not a shape the rules
      allow, and the owner can raise ``max_weight_limit``.

    Deterministic, so every validator bends the same vector the same way.
    """
    shaped = {hotkey: amount for hotkey, amount in units.items() if amount > 0}
    if not shaped:
        raise WeightSubmissionError("there is nothing to send: every allocation is zero")
    if list(shaped) == [validator_hotkey]:
        # A single self-weight is exempt from both limits.
        return ShapedAllocation(units=shaped, padded=(), capped=())
    total = sum(shaped.values())
    padded = _pad_to_minimum(shaped, total, hyperparameters, metagraph, validator_hotkey)
    capped = _cap_to_limit(shaped, total, hyperparameters, metagraph, burn_hotkey)
    return ShapedAllocation(units=shaped, padded=tuple(padded), capped=tuple(capped))


def _pad_to_minimum(
    shaped: dict[str, int],
    total: int,
    hyperparameters: ChainHyperparameters,
    metagraph: MetagraphView,
    validator_hotkey: str,
) -> list[str]:
    required = min(len(metagraph), hyperparameters.min_allowed_weights)
    missing = required - len(shaped)
    if missing <= 0:
        return []
    candidates = [
        hotkey
        for _uid, hotkey in sorted(zip(metagraph.uids, metagraph.hotkeys, strict=True))
        if hotkey not in shaped and hotkey != validator_hotkey
    ]
    if len(candidates) < missing and metagraph.is_registered(validator_hotkey):
        # Only when the subnet is so small that nothing else is left.
        candidates.append(validator_hotkey)
    if len(candidates) < missing:
        raise WeightSubmissionError(
            f"the subnet requires {required} weights and only {len(shaped) + len(candidates)} "
            "uids are available to carry them"
        )
    # The smallest amount that is still at least one u16 unit after scaling.
    pad = -(-total // CHAIN_WEIGHT_UNITS)
    donor = max(shaped, key=lambda hotkey: (shaped[hotkey], hotkey))
    if shaped[donor] - pad * missing < pad:
        raise WeightSubmissionError("no entry is large enough to pay for the padding")
    added = candidates[:missing]
    for hotkey in added:
        shaped[hotkey] = pad
    shaped[donor] -= pad * missing
    return added


def _cap_to_limit(
    shaped: dict[str, int],
    total: int,
    hyperparameters: ChainHyperparameters,
    metagraph: MetagraphView,
    burn_hotkey: str,
) -> list[str]:
    limit = hyperparameters.max_weight_limit
    if limit >= CHAIN_WEIGHT_UNITS:
        return []
    # One u16 unit under the limit absorbs the rounding in the SDK's conversion.
    cap = total * max(limit - 1, 0) // CHAIN_WEIGHT_UNITS
    can_burn = burn_hotkey in shaped or metagraph.is_registered(burn_hotkey)
    holders = len(shaped) + (1 if can_burn and burn_hotkey not in shaped else 0)
    if cap * holders < total:
        raise WeightSubmissionError(
            f"max_weight_limit={limit} lets one uid hold at most {cap} of {total} units, "
            f"and this vector has {holders} possible holders, so it cannot be sent without "
            "paying uids nobody scored. The subnet owner has to raise max_weight_limit"
        )
    capped: list[str] = []
    while over := sorted(hotkey for hotkey, amount in shaped.items() if amount > cap):
        excess = sum(shaped[hotkey] - cap for hotkey in over)
        for hotkey in over:
            shaped[hotkey] = cap
            if hotkey not in capped:
                capped.append(hotkey)
        if can_burn:
            # Emission nobody may be paid is burned, as far as the cap allows.
            burned = min(excess, cap - shaped.get(burn_hotkey, 0))
            if burned > 0:
                shaped[burn_hotkey] = shaped.get(burn_hotkey, 0) + burned
                excess -= burned
        if not excess:
            continue
        under = sorted((hotkey, amount) for hotkey, amount in shaped.items() if amount < cap)
        targets = [
            AllocationTarget(key=hotkey, weight=Fraction(amount)) for hotkey, amount in under
        ]
        for hotkey, extra in allocate(excess, targets).items():
            shaped[hotkey] += extra
    return capped


def sdk_emitted(vector: ChainWeightVector) -> tuple[list[int], list[int]]:
    """The ``(uids, weights)`` the SDK will actually put in the extrinsic.

    Mirrors ``convert_weights_and_uids_for_emit``: scale so the largest weight is
    65535, round each, and drop any that round to zero.
    """
    largest = float(max(vector.weights))
    uids: list[int] = []
    weights: list[int] = []
    for uid, weight in zip(vector.uids, vector.weights, strict=True):
        scaled = round(float(weight) / largest * CHAIN_WEIGHT_UNITS)
        if scaled:
            uids.append(uid)
            weights.append(scaled)
    return uids, weights


def chain_refusal(
    vector: ChainWeightVector,
    *,
    hyperparameters: ChainHyperparameters,
    subnet_n: int,
    validator_uid: int | None,
) -> str | None:
    """Why the chain would refuse this vector, or ``None`` if it would accept it.

    The same two checks subtensor runs in ``do_set_weights``: the minimum
    length, then the largest normalised share against ``max_weight_limit``.
    A single weight on the validator's own uid is exempt from both.
    """
    uids, weights = sdk_emitted(vector)
    if not uids:
        return "every weight rounds to zero"
    if len(uids) == 1 and uids[0] == validator_uid:
        return None
    required = min(subnet_n, hyperparameters.min_allowed_weights)
    if len(uids) < required:
        return f"{len(uids)} weights, and the subnet requires at least {required}"
    limit = hyperparameters.max_weight_limit
    if limit < CHAIN_WEIGHT_UNITS and max(weights) * CHAIN_WEIGHT_UNITS > limit * sum(weights):
        return (
            f"one uid holds {max(weights) / sum(weights):.4%} of the vector, over "
            f"max_weight_limit={limit} ({limit / CHAIN_WEIGHT_UNITS:.4%})"
        )
    return None


__all__ = [
    "CHAIN_WEIGHT_UNITS",
    "ChainCapabilities",
    "ChainHyperparameters",
    "ChainWeightVector",
    "ShapedAllocation",
    "WeightAllocation",
    "assert_submission_allowed",
    "chain_refusal",
    "resolve_allocations",
    "sdk_emitted",
    "shape_for_chain",
    "to_u16_chain_vector",
]
