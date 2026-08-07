"""Model scoring: raw accuracy, the efficiency term, and the spread guard."""

from __future__ import annotations

from fractions import Fraction

import pytest

from prometheon.config import ScoringConfig
from prometheon.errors import ScoringError
from prometheon.scoring.model import (
    ModelEvaluation,
    ModelScore,
    allocate_model_pool,
    rank_models,
    score_models,
)

pytestmark = pytest.mark.unit

DEFAULTS = ScoringConfig()

#: Spread guard off, so the efficiency term always applies.
NO_GUARD = ScoringConfig(efficiency_min_cv_bp=0)

#: Efficiency off entirely.
NO_EFFICIENCY = ScoringConfig(efficiency_lambda_bp=0)


def _evaluation(
    hotkey: str,
    correct: int,
    scored: int,
    *,
    mean_tokens: int = 0,
    readings: int = 1,
) -> ModelEvaluation:
    return ModelEvaluation(
        hotkey=hotkey,
        correct_count=correct,
        scored_count=scored,
        total_tokens=mean_tokens * readings,
        token_reading_count=readings,
    )


def _by_hotkey(ranking: tuple[ModelScore, ...]) -> dict[str, ModelScore]:
    return {entry.hotkey: entry for entry in ranking}


def test_accuracy_is_raw_correct_over_scored() -> None:
    ranking = score_models([_evaluation("hk-a", 803, 1000)], scoring=NO_EFFICIENCY)

    assert ranking.scores[0].accuracy == Fraction(803, 1000)
    assert ranking.scores[0].score == Fraction(803, 1000)


def test_with_efficiency_off_ranking_is_accuracy_alone() -> None:
    ranking = score_models(
        [
            _evaluation("hk-cheap", 800, 1000, mean_tokens=10),
            _evaluation("hk-costly", 900, 1000, mean_tokens=100_000),
        ],
        scoring=NO_EFFICIENCY,
    )

    assert [entry.hotkey for entry in ranking.scores] == ["hk-costly", "hk-cheap"]
    assert ranking.applied_lambda_bp == 0


def test_the_efficiency_term_discounts_the_expensive_model() -> None:
    ranking = score_models(
        [
            _evaluation("hk-cheap", 800, 1000, mean_tokens=100),
            _evaluation("hk-mid", 800, 1000, mean_tokens=200),
            _evaluation("hk-costly", 900, 1000, mean_tokens=1000),
        ],
        scoring=DEFAULTS,
    )

    assert ranking.applied_lambda_bp == 1500
    # P90 of [100, 200, 1000] with linear interpolation: 200 + 0.8 * 800.
    assert ranking.p90_mean_tokens == Fraction(840)

    scores = _by_hotkey(ranking.scores)
    assert scores["hk-costly"].efficiency_t == Fraction(1)  # above P90, clamped
    assert scores["hk-cheap"].efficiency_t == Fraction(100, 840)
    assert scores["hk-costly"].score == Fraction(9, 10) * (1 - Fraction(15, 100))
    # A ten-point accuracy lead is not enough to survive a full 15% discount
    # against a model that is 8x cheaper.
    assert [entry.hotkey for entry in ranking.scores] == ["hk-cheap", "hk-mid", "hk-costly"]


def test_efficiency_never_overturns_an_accuracy_gap_wider_than_lambda() -> None:
    ranking = score_models(
        [
            _evaluation("hk-cheap", 700, 1000, mean_tokens=10),
            _evaluation("hk-costly", 900, 1000, mean_tokens=100_000),
        ],
        scoring=NO_GUARD,
    )

    assert [entry.hotkey for entry in ranking.scores] == ["hk-costly", "hk-cheap"]


def test_the_spread_guard_switches_lambda_off_when_the_field_costs_the_same() -> None:
    # Mean tokens 1000 vs 1020: CV is about 99bp, under the 1000bp threshold.
    evaluations = [
        _evaluation("hk-a", 850, 1000, mean_tokens=1000),
        _evaluation("hk-b", 851, 1000, mean_tokens=1020),
    ]

    guarded = score_models(evaluations, scoring=DEFAULTS)

    assert guarded.applied_lambda_bp == 0
    assert guarded.mean_token_cv_micros == pytest.approx(9_900, abs=100)
    # Untouched accuracy decides it, and hk-b is one item better.
    assert [entry.hotkey for entry in guarded.scores] == ["hk-b", "hk-a"]
    assert guarded.scores[0].score == Fraction(851, 1000)


def test_without_the_guard_that_same_noise_would_have_flipped_the_ranking() -> None:
    evaluations = [
        _evaluation("hk-a", 850, 1000, mean_tokens=1000),
        _evaluation("hk-b", 851, 1000, mean_tokens=1020),
    ]

    unguarded = score_models(evaluations, scoring=NO_GUARD)

    assert unguarded.applied_lambda_bp == 1500
    assert [entry.hotkey for entry in unguarded.scores] == ["hk-a", "hk-b"]


def test_the_guard_still_reports_the_efficiency_it_declined_to_apply() -> None:
    ranking = score_models(
        [
            _evaluation("hk-a", 850, 1000, mean_tokens=1000),
            _evaluation("hk-b", 851, 1000, mean_tokens=1020),
        ],
        scoring=DEFAULTS,
    )

    scores = _by_hotkey(ranking.scores)
    assert scores["hk-b"].efficiency_t == Fraction(1)
    assert scores["hk-a"].efficiency_t == Fraction(1000, 1018)
    # ...but it left the scores alone.
    assert scores["hk-a"].score == scores["hk-a"].accuracy


def test_a_single_model_has_no_spread_so_efficiency_cannot_apply() -> None:
    ranking = score_models([_evaluation("hk-a", 900, 1000, mean_tokens=50_000)], scoring=DEFAULTS)

    assert ranking.applied_lambda_bp == 0
    assert ranking.scores[0].score == Fraction(9, 10)


def test_a_field_with_no_token_readings_at_all_disables_efficiency() -> None:
    ranking = score_models(
        [
            ModelEvaluation(hotkey="hk-a", correct_count=900, scored_count=1000),
            ModelEvaluation(hotkey="hk-b", correct_count=800, scored_count=1000),
        ],
        scoring=NO_GUARD,
    )

    assert ranking.p90_mean_tokens == Fraction(0)
    assert ranking.applied_lambda_bp == 0
    assert ranking.scores[0].score == Fraction(9, 10)


def test_failed_batches_do_not_drag_a_models_token_average_down() -> None:
    # 1000 items scored but only 400 produced a usage reading; the mean is over
    # the readings, not over everything that was attempted.
    ranking = score_models(
        [ModelEvaluation("hk-a", 900, 1000, total_tokens=800_000, token_reading_count=400)],
        scoring=DEFAULTS,
    )

    assert ranking.scores[0].mean_total_tokens == Fraction(2000)


def test_a_model_with_nothing_scored_is_dropped_rather_than_given_an_accuracy() -> None:
    ranking = score_models(
        [_evaluation("hk-silent", 0, 0), _evaluation("hk-a", 500, 1000, mean_tokens=10)],
        scoring=DEFAULTS,
    )

    assert [entry.hotkey for entry in ranking.scores] == ["hk-a"]


def test_a_field_where_nothing_was_scored_ranks_nobody() -> None:
    ranking = score_models([_evaluation("hk-a", 0, 0)], scoring=DEFAULTS)

    assert ranking.scores == ()
    assert ranking.applied_lambda_bp == 0


def test_ties_break_on_accuracy_then_hotkey_never_on_input_order() -> None:
    tied = [
        ModelScore(
            hotkey=hotkey,
            correct_count=correct,
            scored_count=1000,
            accuracy=Fraction(correct, 1000),
            mean_total_tokens=Fraction(0),
            efficiency_t=Fraction(0),
            score=Fraction(1, 2),
        )
        for hotkey, correct in (("hk-z", 900), ("hk-a", 500), ("hk-b", 900))
    ]

    ranked = [entry.hotkey for entry in rank_models(tied)]

    assert ranked == ["hk-b", "hk-z", "hk-a"]
    assert ranked == [entry.hotkey for entry in rank_models(list(reversed(tied)))]


def test_more_correct_than_scored_is_refused() -> None:
    with pytest.raises(ScoringError, match="more correct items than scored"):
        score_models([_evaluation("hk-a", 11, 10)], scoring=DEFAULTS)


def test_negative_counts_are_refused() -> None:
    with pytest.raises(ScoringError, match="negative evaluation counts"):
        score_models(
            [ModelEvaluation("hk-a", 0, 10, total_tokens=-1, token_reading_count=1)],
            scoring=DEFAULTS,
        )


def test_a_repeated_hotkey_is_refused() -> None:
    with pytest.raises(ScoringError, match="duplicate"):
        score_models([_evaluation("hk-a", 1, 10), _evaluation("hk-a", 2, 10)], scoring=DEFAULTS)


def test_three_models_take_the_designed_thirty_fifteen_five_split() -> None:
    ranking = score_models(
        [
            _evaluation("hk-1", 900, 1000, mean_tokens=10),
            _evaluation("hk-2", 800, 1000, mean_tokens=10),
            _evaluation("hk-3", 700, 1000, mean_tokens=10),
        ],
        scoring=DEFAULTS,
    )

    units = allocate_model_pool(
        ranking.scores, pool=500_000, rank_shares_bp=DEFAULTS.model_rank_shares_bp
    )

    assert units == {"hk-1": 300_000, "hk-2": 150_000, "hk-3": 50_000}
    assert sum(units.values()) == 500_000


def test_two_models_renormalise_to_three_thousand_over_forty_five_hundred() -> None:
    ranking = score_models(
        [
            _evaluation("hk-1", 900, 1000, mean_tokens=10),
            _evaluation("hk-2", 800, 1000, mean_tokens=10),
        ],
        scoring=DEFAULTS,
    )

    units = allocate_model_pool(
        ranking.scores, pool=450_000, rank_shares_bp=DEFAULTS.model_rank_shares_bp
    )

    assert units == {"hk-1": 300_000, "hk-2": 150_000}
    assert sum(units.values()) == 450_000


def test_one_model_takes_the_whole_model_pool() -> None:
    ranking = score_models([_evaluation("hk-1", 1, 1000, mean_tokens=10)], scoring=DEFAULTS)

    units = allocate_model_pool(
        ranking.scores, pool=500_000, rank_shares_bp=DEFAULTS.model_rank_shares_bp
    )

    assert units == {"hk-1": 500_000}


def test_a_fourth_place_model_is_not_paid() -> None:
    ranking = score_models(
        [_evaluation(f"hk-{index}", 900 - index, 1000, mean_tokens=10) for index in range(1, 6)],
        scoring=DEFAULTS,
    )

    units = allocate_model_pool(
        ranking.scores, pool=500_000, rank_shares_bp=DEFAULTS.model_rank_shares_bp
    )

    assert list(units) == ["hk-1", "hk-2", "hk-3"]
    assert sum(units.values()) == 500_000


def test_no_models_allocate_nothing_so_the_pool_can_burn() -> None:
    assert allocate_model_pool([], pool=500_000, rank_shares_bp=(3000, 1500, 500)) == {}


def test_model_allocation_does_not_trust_the_order_it_was_handed() -> None:
    ranking = score_models(
        [
            _evaluation("hk-1", 900, 1000, mean_tokens=10),
            _evaluation("hk-2", 800, 1000, mean_tokens=10),
        ],
        scoring=DEFAULTS,
    )

    shuffled = allocate_model_pool(
        list(reversed(ranking.scores)), pool=450_000, rank_shares_bp=(3000, 1500, 500)
    )

    assert shuffled == {"hk-1": 300_000, "hk-2": 150_000}


def test_rank_shares_that_pay_nobody_allocate_nothing() -> None:
    ranking = score_models([_evaluation("hk-1", 1, 10, mean_tokens=10)], scoring=DEFAULTS)

    assert allocate_model_pool(ranking.scores, pool=500_000, rank_shares_bp=(0, 0)) == {}


def test_empty_rank_shares_are_refused() -> None:
    with pytest.raises(ScoringError, match="rank shares are empty"):
        allocate_model_pool([], pool=500_000, rank_shares_bp=())


def test_negative_rank_shares_are_refused() -> None:
    with pytest.raises(ScoringError, match="must not be negative"):
        allocate_model_pool([], pool=500_000, rank_shares_bp=(3000, -1))


def test_a_negative_model_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="must not be negative"):
        allocate_model_pool([], pool=-1, rank_shares_bp=(3000,))


class TestAZeroScoreIsNeverPaid:
    """Ranking top-3 is not the same as having earned anything.

    Without this filter the model half could never burn, contradicting both
    ``docs/scoring.md`` and the dataset half, which has always filtered.
    """

    def test_a_model_that_answered_nothing_earns_nothing(self) -> None:
        """A chute returning HTTP 500 to every batch scores exactly zero."""
        dead = [ModelEvaluation(hotkey="hk-broken", correct_count=0, scored_count=1000)]
        allocation = allocate_model_pool(
            score_models(dead, scoring=DEFAULTS).scores,
            pool=500_000,
            rank_shares_bp=(3000, 1500, 500),
        )
        assert allocation == {}

    def test_a_dead_chute_does_not_take_emission_off_the_honest_field(self) -> None:
        """It used to collect 500bp of total miner emission for doing nothing."""
        field = [
            ModelEvaluation("hk-a", 850, 1000, 100_000, 1000),
            ModelEvaluation("hk-b", 851, 1000, 102_000, 1000),
            ModelEvaluation("hk-grief", 0, 1000),
        ]
        allocation = allocate_model_pool(
            score_models(field, scoring=DEFAULTS).scores,
            pool=500_000,
            rank_shares_bp=(3000, 1500, 500),
        )
        assert "hk-grief" not in allocation
        assert sum(allocation.values()) == 500_000
        assert set(allocation) == {"hk-a", "hk-b"}

    def test_an_all_zero_field_burns_rather_than_paying_alphabetically(self) -> None:
        """The worst version of the bug.

        With every score zero, ranking falls through to the hotkey tie-break,
        so half of miner emission would be handed out in lexicographic order,
        identically on every validator, so consensus cannot correct it.
        """
        field = [
            ModelEvaluation(hotkey, 0, 1000)
            for hotkey in ("hk-zulu", "hk-alpha", "hk-mike", "hk-bravo")
        ]
        allocation = allocate_model_pool(
            score_models(field, scoring=DEFAULTS).scores,
            pool=500_000,
            rank_shares_bp=(3000, 1500, 500),
        )
        assert allocation == {}


class TestUnmeasuredModelsDoNotSteerTheField:
    """A model with no token reading must not vote on the token scale.

    A mean of zero says nothing about cost, because there was no measurement.
    One such entry could drag the P90 down and inflate the spread past the CV
    guard, switching lambda back on across a field the guard had correctly
    called noise. That is enough to swap rank 1 and rank 2 and move 1500bp of
    total emission.
    """

    HONEST = (
        ModelEvaluation("hk-a", 850, 1000, 1_000_000, 1000),
        ModelEvaluation("hk-b", 851, 1000, 1_020_000, 1000),
        ModelEvaluation("hk-c", 840, 1000, 1_010_000, 1000),
    )

    def test_a_griefer_does_not_reorder_the_honest_field(self) -> None:
        alone = score_models(self.HONEST, scoring=DEFAULTS)
        with_griefer = score_models(
            [*self.HONEST, ModelEvaluation("hk-grief", 0, 1000)], scoring=DEFAULTS
        )

        assert [entry.hotkey for entry in alone.scores] == [
            entry.hotkey for entry in with_griefer.scores if entry.hotkey != "hk-grief"
        ]

    def test_a_griefer_does_not_switch_the_spread_guard_back_on(self) -> None:
        alone = score_models(self.HONEST, scoring=DEFAULTS)
        with_griefer = score_models(
            [*self.HONEST, ModelEvaluation("hk-grief", 0, 1000)], scoring=DEFAULTS
        )
        assert alone.applied_lambda_bp == 0
        assert with_griefer.applied_lambda_bp == alone.applied_lambda_bp

    def test_a_griefer_does_not_collapse_the_p90(self) -> None:
        """A dragged-down P90 clamps every honest model to the full discount."""
        alone = score_models(self.HONEST, scoring=NO_GUARD)
        with_griefer = score_models(
            [*self.HONEST, ModelEvaluation("hk-grief", 0, 1000)], scoring=NO_GUARD
        )
        assert alone.p90_mean_tokens == with_griefer.p90_mean_tokens

    def test_no_reading_is_the_worst_efficiency_not_the_best(self) -> None:
        """Otherwise omitting usage figures is a strategy.

        Reporting nothing must never be cheaper than reporting honestly, or
        every model has an incentive to stop reporting.
        """
        ranking = score_models(
            [*self.HONEST, ModelEvaluation("hk-silent", 900, 1000)], scoring=NO_GUARD
        )
        silent = next(entry for entry in ranking.scores if entry.hotkey == "hk-silent")
        assert silent.mean_total_tokens is None
        assert silent.efficiency_t == 1
        assert all(
            entry.efficiency_t <= silent.efficiency_t
            for entry in ranking.scores
            if entry.hotkey != "hk-silent"
        )


class TestAModelMustBeMeasuredBeforeItIsPaid:
    """Accuracy over one item is 1 or 0, and neither means anything.

    A miner can arrange this. Its own test content is excluded from the corpus
    it is scored on, so a miner supplying most of a day's test content is
    scored on very little of it.
    """

    def test_a_one_item_measurement_cannot_outrank_a_five_thousand_item_one(self) -> None:
        field = [
            ModelEvaluation("hk-flood", 1, 1),
            ModelEvaluation("hk-careful", 4999, 5000, 500_000, 5000),
        ]
        allocation = allocate_model_pool(
            score_models(field, scoring=DEFAULTS).scores,
            pool=500_000,
            rank_shares_bp=(3000, 1500, 500),
        )
        assert allocation == {"hk-careful": 500_000}

    def test_the_floor_is_configurable(self) -> None:
        field = [ModelEvaluation("hk-small", 40, 50, 5_000, 50)]
        assert score_models(field, scoring=ScoringConfig(model_min_scored_items=10)).scores
        assert not score_models(field, scoring=ScoringConfig(model_min_scored_items=51)).scores

    def test_an_under_measured_model_does_not_vote_on_the_token_scale(self) -> None:
        """Excluded means excluded, field statistics included."""
        honest = [
            ModelEvaluation("hk-a", 850, 1000, 1_000_000, 1000),
            ModelEvaluation("hk-b", 851, 1000, 1_020_000, 1000),
        ]
        alone = score_models(honest, scoring=NO_GUARD)
        with_tiny = score_models(
            [*honest, ModelEvaluation("hk-tiny", 1, 1, 999_999, 1)], scoring=NO_GUARD
        )
        assert alone.p90_mean_tokens == with_tiny.p90_mean_tokens
