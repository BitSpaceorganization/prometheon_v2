"""The published rules: every value that decides who is paid, as code.

These used to be config, in ``[scoring]`` and ``[evaluation]``. A config value
is the operator's own copy of a rule. A release that changed one also needed
every operator to edit their file, and an operator who didn't went on scoring
under the old number, disagreeing with everyone else for a reason no log line
named. As code, a release carries its rules with it, so an update never needs a
config edit. A config file that still sets them loads, and says it ignored them
(:data:`prometheon.config.RETIRED_SETTINGS`).

Every validator applies the same values. Changing one ships as a new
``SCORING_VERSION``; CONTRIBUTING.md lists the steps. The other published rules
live beside the code that applies them: the submission fee in
:mod:`prometheon.version`, the parameter cap and the size penalty in
:mod:`prometheon.scoring.size`, and the ranking band in
:mod:`prometheon.scoring.model`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from prometheon.errors import ScoringError

_BASIS_POINTS: Final[int] = 10_000


@dataclass(frozen=True)
class ScoringRules:
    """One rule set. :data:`RULES` is the published one.

    A type rather than loose constants so a test can state the rules it
    exercises, and so a cycle is handed its rules instead of reading them from
    somewhere a caller can't see.
    """

    #: The data contribution pool, as a share of **total** emission: 70%.
    #:
    #: It starts high because inviting real people and getting them to write
    #: test content costs far more today than training a text model. As user
    #: numbers grow and model requirements get harder, the share moving to model
    #: training is expected to reach 80 to 90%, as a new scoring version.
    dataset_share_bp: int = 7000

    #: Only the top contributors are paid, proportionally to score. With fewer
    #: than this many scoring above zero, the pool divides among those present.
    dataset_top_n: int = 10

    #: The model pool: a share of **total** emission per place, 25%, 4% and 1%.
    #: A place nobody fills goes to the lowest-placed winner present: two models
    #: take 25% and 5%, one model takes all 30%. The pool burns only when no
    #: model qualifies.
    model_rank_shares_bp: tuple[int, ...] = (2500, 400, 100)

    #: The entry gate, for both pools: a miner counts only with at least this
    #: many identity-verified active members in its group ...
    min_verified_active_members: int = 10

    #: ... and at least this many members in total. Every member is
    #: phone-verified, and the total includes the verified active ones. Both
    #: counts come from the signed epoch.
    min_members: int = 30

    #: The smallest number of scored items a model may hold a place on.
    #:
    #: Accuracy over one item is 100% or 0% and means nothing either way, but
    #: it sorts above a model that got 4999 of 5000 right. That is reachable: a
    #: miner's own test content is excluded from its corpus, so a miner who
    #: supplies almost all of a day's test content is scored on almost nothing.
    model_min_scored_items: int = 100

    #: Seconds one model has to load and judge its whole corpus. A model that
    #: has not finished gets no score from this validator for the day, and its
    #: unfinished items count for nothing (B28). A model at the 4-billion
    #: parameter cap needs minutes on one 24 GB card; the margin keeps two
    #: validators on different cards from disagreeing about an honest model.
    model_timeout_seconds: float = 1800.0

    #: Items a model judges per engine call, and so how many a failed call
    #: costs: a batch that fails counts every item in it wrong. Every validator
    #: has to cut the corpus the same way for a failure to cost the same.
    evaluation_batch_size: int = 100

    #: Ceiling on a model's weights: 16 GiB holds a model at the parameter cap
    #: in 32-bit floats, loaded in 16-bit on a 24 GB card. The hub lists only
    #: models under the same ceiling, so a validator on another number would
    #: evaluate models the hub never charged for, or the reverse.
    max_weight_bytes: int = 16 * 1024**3

    #: Ceiling on everything in a repository that isn't weights: config,
    #: tokenizer, README. Every validator downloads all of it.
    max_non_weight_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        if not 0 <= self.dataset_share_bp <= _BASIS_POINTS:
            raise ScoringError(
                f"dataset_share_bp must be within 0..10000, got {self.dataset_share_bp}"
            )
        if self.dataset_top_n < 1:
            raise ScoringError(f"dataset_top_n must be at least 1, got {self.dataset_top_n}")
        shares = self.model_rank_shares_bp
        if not shares or any(share < 0 for share in shares) or sum(shares) == 0:
            raise ScoringError(f"model rank shares must be non-negative and not all zero: {shares}")
        if self.dataset_share_bp + sum(shares) > _BASIS_POINTS:
            raise ScoringError(
                "the data share and the model shares together are over 100% of emission; "
                "they are shares of the total, not ratios"
            )
        if self.min_verified_active_members < 0 or self.min_members < 0:
            raise ScoringError("membership floors cannot be negative")
        if self.model_min_scored_items < 1:
            raise ScoringError("model_min_scored_items must be at least 1")
        if self.model_timeout_seconds <= 0:
            raise ScoringError("model_timeout_seconds must be positive")
        if not 1 <= self.evaluation_batch_size <= 100:
            raise ScoringError("evaluation_batch_size must be within 1..100")
        if self.max_weight_bytes <= 0 or self.max_non_weight_bytes <= 0:
            raise ScoringError("the byte ceilings must be positive")


#: The rules this release scores under.
RULES: Final[ScoringRules] = ScoringRules()

__all__ = ["RULES", "ScoringRules"]
