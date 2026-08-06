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
from typing import Final

from prometheon.chain.metagraph import MetagraphView
from prometheon.errors import (
    CommitRevealEnabledError,
    DuplicateWeightTargetError,
    MechidMissingError,
    WeightSubmissionError,
    WeightsVersionMismatchError,
)

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
            "[chain] allow_missing_mechid = true — which is for localnet only, "
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


__all__ = [
    "CHAIN_WEIGHT_UNITS",
    "ChainCapabilities",
    "ChainHyperparameters",
    "ChainWeightVector",
    "WeightAllocation",
    "assert_submission_allowed",
    "resolve_allocations",
    "to_u16_chain_vector",
]
