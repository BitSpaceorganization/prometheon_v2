"""Model scoring: accuracy less a size penalty, and a ranking that favours the first.

Every expected number is worked by hand in the test or its docstring, from the
rules in docs/scoring.md:

    raw_accuracy_bp = correct * 10,000 / scored, rounded half up
    size_penalty_bp = floor(100 * log2(P / 500M)), 0 up to 500M
    model_score_bp  = max(0, raw_accuracy_bp - size_penalty_bp)

and each place goes to the earliest commit among the models within 100 bp of
the best remaining score.
"""

from __future__ import annotations

import pytest

from prometheon.errors import ScoringError
from prometheon.rules import RULES, ScoringRules
from prometheon.scoring.model import (
    RANK_BAND_BP,
    ModelEvaluation,
    ModelRanking,
    ModelScore,
    accuracy_bp,
    allocate_model_pool,
    rank_models,
    score_models,
)

pytestmark = pytest.mark.unit

#: The day's corpus here is a thousand items, well above the shipped floor.
DEFAULTS = RULES
TOTAL = 1_000_000
SHARES = (2500, 400, 100)
SMALL = 400_000_000  # under 500M parameters: no penalty


def _evaluation(
    hotkey: str,
    correct: int,
    *,
    scored: int = 1000,
    block: int = 100,
    parameters: int = SMALL,
    timed_out: bool = False,
) -> ModelEvaluation:
    return ModelEvaluation(
        hotkey=hotkey,
        correct_count=correct,
        scored_count=scored,
        commit_block=block,
        parameters=parameters,
        timed_out=timed_out,
    )


def _ranking(*evaluations: ModelEvaluation, rules: ScoringRules = DEFAULTS) -> ModelRanking:
    return score_models(list(evaluations), rules=rules)


def _order(ranking: ModelRanking) -> list[str]:
    return [score.hotkey for score in ranking.ranked]


def _score(ranking: ModelRanking, hotkey: str) -> ModelScore:
    return next(score for score in ranking.scores if score.hotkey == hotkey)


# --- the score ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("correct", "scored", "expected"),
    [
        (900, 1000, 9000),
        (2, 3, 6667),  # 6666.67 rounds up
        (1, 3, 3333),  # 3333.33 rounds down
        (1, 8, 1250),
        (1, 16_000, 1),  # 0.625 rounds up
        (0, 10, 0),
        (10, 10, 10_000),
        (0, 0, 0),
    ],
)
def test_raw_accuracy_is_basis_points_rounded_half_up(
    correct: int, scored: int, expected: int
) -> None:
    assert accuracy_bp(correct, scored) == expected


def test_the_score_is_accuracy_less_the_size_penalty() -> None:
    """2B parameters: floor(100 * log2(4)) = 200 bp off 9,000."""
    ranking = _ranking(_evaluation("hk-a", 900, parameters=2_000_000_000))
    score = _score(ranking, "hk-a")

    assert (score.raw_accuracy_bp, score.size_penalty_bp, score.score_bp) == (9000, 200, 8800)


def test_a_model_at_or_under_half_a_billion_parameters_pays_nothing() -> None:
    ranking = _ranking(_evaluation("hk-a", 900, parameters=500_000_000))

    assert _score(ranking, "hk-a").score_bp == 9000


def test_the_score_never_goes_below_zero() -> None:
    """1% accuracy at 4B parameters: 100 - 300 would be negative; it is 0."""
    ranking = _ranking(_evaluation("hk-a", 10, parameters=4_000_000_000))

    assert _score(ranking, "hk-a").score_bp == 0
    assert ranking.ranked == ()


def test_a_bigger_model_must_be_more_accurate_to_rank_level_with_a_smaller_one() -> None:
    """Big: 9,300 - floor(100 * log2(8)) = 9,000. Small: 9,250 - 0 = 9,250.

    Small is best, and big is 250 bp behind, outside the band: small first.
    """
    ranking = _ranking(
        _evaluation("hk-big", 930, block=100, parameters=4_000_000_000),
        _evaluation("hk-small", 925, block=200),
    )

    assert _order(ranking) == ["hk-small", "hk-big"]


def test_a_model_that_timed_out_has_no_score_and_does_not_rank() -> None:
    ranking = _ranking(
        _evaluation("hk-slow", 1000, timed_out=True),
        _evaluation("hk-a", 800),
    )

    assert [score.hotkey for score in ranking.scores] == ["hk-a"]
    assert _order(ranking) == ["hk-a"]
    assert ranking.rank_of("hk-slow") is None


def test_a_model_with_nothing_scored_is_dropped_rather_than_given_an_accuracy() -> None:
    ranking = _ranking(_evaluation("hk-empty", 0, scored=0), _evaluation("hk-a", 800))

    assert [score.hotkey for score in ranking.scores] == ["hk-a"]


# --- ranking: the first committed keeps its place within a point --------------


def test_an_earlier_model_keeps_first_place_against_a_later_one_within_a_point() -> None:
    """Early: 9,000. Late: 9,099, 99 bp better. 9,000 >= 9,099 - 100, so the
    earlier commit wins first place, and the later one is second."""
    ranking = _ranking(
        _evaluation("hk-late", 9099, scored=10_000, block=500),
        _evaluation("hk-early", 9000, scored=10_000, block=100),
    )

    assert _order(ranking) == ["hk-early", "hk-late"]


def test_exactly_one_point_better_is_not_enough() -> None:
    """The band is inclusive: 9,000 >= 9,100 - 100."""
    ranking = _ranking(
        _evaluation("hk-late", 9100, scored=10_000, block=500),
        _evaluation("hk-early", 9000, scored=10_000, block=100),
    )

    assert _order(ranking) == ["hk-early", "hk-late"]


def test_beating_it_by_more_than_a_full_point_takes_the_place() -> None:
    """9,000 < 9,101 - 100: the earlier model is outside the band."""
    ranking = _ranking(
        _evaluation("hk-late", 9101, scored=10_000, block=500),
        _evaluation("hk-early", 9000, scored=10_000, block=100),
    )

    assert _order(ranking) == ["hk-late", "hk-early"]


def test_the_band_is_one_point() -> None:
    assert RANK_BAND_BP == 100


def test_each_place_is_decided_again_among_the_models_left() -> None:
    """A 9,500 at block 300, B 9,450 at block 100, C 9,300 at block 200.

    First: best 9,500, band >= 9,400 holds A and B; B committed first.
    Second: best of A and C is 9,500, band >= 9,400 holds only A.
    Third: C.
    """
    ranking = _ranking(
        _evaluation("hk-a", 9500, scored=10_000, block=300),
        _evaluation("hk-b", 9450, scored=10_000, block=100),
        _evaluation("hk-c", 9300, scored=10_000, block=200),
    )

    assert _order(ranking) == ["hk-b", "hk-a", "hk-c"]


def test_the_band_is_measured_from_the_best_not_chained_through_neighbours() -> None:
    """A 9,000 at block 1, B 9,090 at block 2, C 9,180 at block 3.

    A is within a point of B, and B of C, but A is 180 bp behind C, the best.
    First: band >= 9,080 holds B and C; B committed first. Second: band >= 9,080
    over A and C holds only C. Third: A.
    """
    ranking = _ranking(
        _evaluation("hk-a", 9000, scored=10_000, block=1),
        _evaluation("hk-b", 9090, scored=10_000, block=2),
        _evaluation("hk-c", 9180, scored=10_000, block=3),
    )

    assert _order(ranking) == ["hk-b", "hk-c", "hk-a"]


def test_the_same_commit_block_is_settled_by_hotkey() -> None:
    ranking = _ranking(
        _evaluation("hk-b", 900, block=100),
        _evaluation("hk-a", 900, block=100),
    )

    assert _order(ranking) == ["hk-a", "hk-b"]


def test_the_size_penalty_counts_before_the_band_is_drawn() -> None:
    """Big: 9,350 raw at 2B parameters, 9,150 after the penalty, block 100.
    Small: 9,200 raw, no penalty, block 200. Big is within a point of small's
    9,200 and committed first, so it keeps first place."""
    ranking = _ranking(
        _evaluation("hk-big", 935, block=100, parameters=2_000_000_000),
        _evaluation("hk-small", 920, block=200),
    )

    assert _order(ranking) == ["hk-big", "hk-small"]
    assert _score(ranking, "hk-big").score_bp == 9150


def test_the_ranking_does_not_depend_on_the_order_models_arrive_in() -> None:
    field = [
        _evaluation("hk-a", 9500, scored=10_000, block=300),
        _evaluation("hk-b", 9450, scored=10_000, block=100),
        _evaluation("hk-c", 9300, scored=10_000, block=200),
        _evaluation("hk-d", 9300, scored=10_000, block=200),
    ]

    forward = _order(_ranking(*field))
    backward = _order(_ranking(*reversed(field)))

    assert forward == backward == ["hk-b", "hk-a", "hk-c", "hk-d"]


class TestAModelMustBeMeasuredBeforeItRanks:
    """A model scored on fewer than the minimum items is not ranked."""

    def test_a_one_item_measurement_cannot_outrank_a_thousand_item_one(self) -> None:
        ranking = _ranking(
            _evaluation("hk-lucky", 1, scored=1, block=1),
            _evaluation("hk-honest", 999, block=2),
        )

        assert _order(ranking) == ["hk-honest"]
        assert _score(ranking, "hk-lucky").score_bp == 10_000

    def test_the_floor_is_the_configured_one(self) -> None:
        ranking = _ranking(
            _evaluation("hk-a", 9, scored=10),
            rules=ScoringRules(model_min_scored_items=10),
        )

        assert _order(ranking) == ["hk-a"]


def test_more_correct_than_scored_is_refused() -> None:
    with pytest.raises(ScoringError, match="more correct"):
        _ranking(_evaluation("hk-a", 11, scored=10))


@pytest.mark.parametrize(
    "evaluation",
    [
        _evaluation("hk-a", -1),
        _evaluation("hk-a", 1, scored=-1),
        _evaluation("hk-a", 1, block=-1),
        _evaluation("hk-a", 1, parameters=-1),
    ],
)
def test_negative_inputs_are_refused(evaluation: ModelEvaluation) -> None:
    with pytest.raises(ScoringError, match="negative"):
        _ranking(evaluation)


def test_a_repeated_hotkey_is_refused() -> None:
    with pytest.raises(ScoringError, match="duplicate"):
        _ranking(_evaluation("hk-a", 1), _evaluation("hk-a", 2))


# --- the model pool -------------------------------------------------------------


def _pool(*evaluations: ModelEvaluation, shares: tuple[int, ...] = SHARES) -> dict[str, int]:
    return allocate_model_pool(
        _ranking(*evaluations).scores,
        total_units=TOTAL,
        rank_shares_bp=shares,
        min_scored_items=DEFAULTS.model_min_scored_items,
    )


def test_three_models_take_twenty_five_four_and_one_percent_of_total() -> None:
    assert _pool(
        _evaluation("hk-a", 900, block=1),
        _evaluation("hk-b", 800, block=2),
        _evaluation("hk-c", 700, block=3),
    ) == {"hk-a": 250_000, "hk-b": 40_000, "hk-c": 10_000}


def test_two_models_take_twenty_five_and_five_percent() -> None:
    assert _pool(_evaluation("hk-a", 900, block=1), _evaluation("hk-b", 800, block=2)) == {
        "hk-a": 250_000,
        "hk-b": 50_000,
    }


def test_one_model_takes_the_whole_thirty_percent() -> None:
    assert _pool(_evaluation("hk-a", 900)) == {"hk-a": 300_000}


def test_a_fourth_place_model_is_not_paid() -> None:
    paid = _pool(
        _evaluation("hk-a", 900, block=1),
        _evaluation("hk-b", 800, block=2),
        _evaluation("hk-c", 700, block=3),
        _evaluation("hk-d", 600, block=4),
    )

    assert "hk-d" not in paid
    assert sum(paid.values()) == 300_000


def test_no_models_allocate_nothing_so_the_shares_can_burn() -> None:
    assert _pool() == {}


def test_the_pool_follows_the_band_rule() -> None:
    """The later model is 99 bp better and still second: 25% to the earlier."""
    paid = _pool(
        _evaluation("hk-late", 9099, scored=10_000, block=500),
        _evaluation("hk-early", 9000, scored=10_000, block=100),
    )

    assert paid == {"hk-early": 250_000, "hk-late": 50_000}


def test_model_allocation_does_not_trust_the_order_it_was_handed() -> None:
    scores = _ranking(
        _evaluation("hk-a", 900, block=1),
        _evaluation("hk-b", 800, block=2),
        _evaluation("hk-c", 700, block=3),
    ).scores
    paid = allocate_model_pool(
        tuple(reversed(scores)),
        total_units=TOTAL,
        rank_shares_bp=SHARES,
        min_scored_items=DEFAULTS.model_min_scored_items,
    )

    assert paid == {"hk-a": 250_000, "hk-b": 40_000, "hk-c": 10_000}


def test_a_zero_share_place_pays_nothing() -> None:
    paid = _pool(
        _evaluation("hk-a", 900, block=1),
        _evaluation("hk-b", 800, block=2),
        shares=(3000, 0, 0),
    )

    assert paid == {"hk-a": 300_000}


@pytest.mark.parametrize("shares", [(), (2500, -1)])
def test_unpayable_rank_shares_are_refused(shares: tuple[int, ...]) -> None:
    with pytest.raises(ScoringError):
        allocate_model_pool((), total_units=TOTAL, rank_shares_bp=shares, min_scored_items=1)


def test_a_negative_weight_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="negative"):
        allocate_model_pool((), total_units=-1, rank_shares_bp=SHARES, min_scored_items=1)


class TestAZeroScoreIsNeverPaid:
    """A model that answered nothing all day scores exactly zero."""

    def test_a_model_that_answered_nothing_earns_nothing(self) -> None:
        assert _pool(_evaluation("hk-dead", 0)) == {}

    def test_a_dead_model_does_not_take_emission_off_the_honest_field(self) -> None:
        paid = _pool(_evaluation("hk-dead", 0, block=1), _evaluation("hk-a", 900, block=2))

        assert paid == {"hk-a": 300_000}

    def test_an_all_zero_field_burns_rather_than_paying_alphabetically(self) -> None:
        paid = _pool(
            _evaluation("hk-a", 0, block=1),
            _evaluation("hk-b", 0, block=2),
            _evaluation("hk-c", 0, block=3),
        )

        assert paid == {}

    def test_a_zero_scorer_does_not_hold_a_place_that_rolls_down(self) -> None:
        """Two real models and a dead one: the dead one is not third, so the
        empty third place rolls down to the second, 4% + 1%."""
        paid = _pool(
            _evaluation("hk-a", 900, block=1),
            _evaluation("hk-b", 800, block=2),
            _evaluation("hk-dead", 0, block=0),
        )

        assert paid == {"hk-a": 250_000, "hk-b": 50_000}


def test_rank_models_ranks_only_what_qualifies() -> None:
    scores = _ranking(
        _evaluation("hk-a", 900),
        _evaluation("hk-zero", 0, block=1),
        _evaluation("hk-few", 5, scored=5, block=2),
    ).scores

    assert [score.hotkey for score in rank_models(scores, min_scored_items=100)] == ["hk-a"]
