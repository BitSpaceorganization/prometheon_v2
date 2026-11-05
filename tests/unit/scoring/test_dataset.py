"""Data contribution: 2 * min(V_pos, V_neg) * (V_pos + V_neg) / S, the top-10 cut.

Every expected value here is worked by hand from the formula in the relaunch
spec, so a test failing means the code and the published rule disagree.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from prometheon.errors import ScoringError
from prometheon.scoring.dataset import (
    DatasetScore,
    DatasetSubmission,
    allocate_dataset_pool,
    score_dataset,
)

pytestmark = pytest.mark.unit


def _submission(hotkey: str, submitted: int, v_pos: int, v_neg: int) -> DatasetSubmission:
    return DatasetSubmission(
        hotkey=hotkey,
        submitted_count=submitted,
        violating_confirmed=v_pos,
        compliant_confirmed=v_neg,
    )


class TestTheFormula:
    def test_a_balanced_fully_confirmed_day_scores_its_size(self) -> None:
        # amount = 2 * min(10, 10) = 20, quality = 20/20 = 1.
        (score,) = score_dataset([_submission("hk-a", 20, 10, 10)])
        assert score.score == Fraction(20)

    def test_the_smaller_class_sets_the_amount(self) -> None:
        # amount = 2 * min(30, 10) = 20, quality = 40/40 = 1.
        (score,) = score_dataset([_submission("hk-a", 40, 30, 10)])
        assert score.score == Fraction(20)

    def test_unconfirmed_items_lower_quality(self) -> None:
        # amount = 2 * min(10, 10) = 20, quality = 20/100. Score 4.
        (score,) = score_dataset([_submission("hk-a", 100, 10, 10)])
        assert score.score == Fraction(4)

    def test_the_score_is_an_exact_fraction(self) -> None:
        # amount = 2 * min(1, 1) = 2, quality = 2/3. Score 4/3, not 1.333.
        (score,) = score_dataset([_submission("hk-a", 3, 1, 1)])
        assert score.score == Fraction(4, 3)

    def test_nothing_submitted_scores_zero_without_dividing_by_zero(self) -> None:
        (score,) = score_dataset([_submission("hk-a", 0, 0, 0)])
        assert score.score == 0

    def test_valid_count_is_both_classes(self) -> None:
        (score,) = score_dataset([_submission("hk-a", 9, 4, 3)])
        assert score.valid_count == 7


class TestFloodsEarnNothing:
    """The reason the old ``V^2/S`` was replaced."""

    def test_a_flood_of_confirmed_compliant_posts_scores_zero(self) -> None:
        """B8: 5,000 obviously compliant posts, all confirmed, used to score 5,000."""
        (score,) = score_dataset([_submission("hk-flood", 5000, 0, 5000)])
        assert score.score == 0

    def test_a_flood_of_confirmed_violating_posts_scores_zero(self) -> None:
        (score,) = score_dataset([_submission("hk-flood", 5000, 5000, 0)])
        assert score.score == 0

    def test_a_flood_beside_a_few_pairs_pays_only_for_the_pairs_and_dilutes_them(self) -> None:
        # Five honest pairs plus 990 one-sided items: amount = 2 * 5 = 10,
        # quality = 1000/1000, so the score is 10, exactly what the pairs alone
        # earn. Adding the flood bought nothing.
        flooded = score_dataset([_submission("hk-a", 1000, 995, 5)])[0].score
        honest = score_dataset([_submission("hk-b", 10, 5, 5)])[0].score
        assert flooded == honest == 10

    def test_a_careful_contributor_outranks_a_flooder(self) -> None:
        scores = score_dataset(
            [
                _submission("hk-flooder", 5000, 50, 4950),
                _submission("hk-careful", 120, 60, 55),
            ]
        )
        # flooder: 2 * 50 * 5000/5000 = 100. careful: 2 * 55 * 115/120 = 6325/60.
        assert [entry.hotkey for entry in scores] == ["hk-careful", "hk-flooder"]
        assert scores[0].score == Fraction(2 * 55 * 115, 120)
        assert scores[1].score == Fraction(100)


class TestRefusals:
    def test_more_confirmed_than_submitted_is_refused(self) -> None:
        with pytest.raises(ScoringError, match="more confirmed items than submitted"):
            score_dataset([_submission("hk-a", 10, 6, 5)])

    @pytest.mark.parametrize("counts", [(-1, 0, 0), (1, -1, 0), (1, 0, -1)])
    def test_negative_counts_are_refused(self, counts: tuple[int, int, int]) -> None:
        with pytest.raises(ScoringError, match="negative dataset counts"):
            score_dataset([_submission("hk-a", *counts)])

    def test_a_repeated_hotkey_is_refused_rather_than_scored_twice(self) -> None:
        with pytest.raises(ScoringError, match="duplicate"):
            score_dataset([_submission("hk-a", 10, 5, 5), _submission("hk-a", 10, 5, 5)])


class TestRanking:
    def test_equal_scores_rank_by_hotkey_ascending(self) -> None:
        scores = score_dataset(
            [_submission(hotkey, 20, 10, 10) for hotkey in ("hk-z", "hk-a", "hk-m")]
        )
        assert [entry.hotkey for entry in scores] == ["hk-a", "hk-m", "hk-z"]


class TestThePool:
    def test_the_pool_splits_in_proportion_to_score(self) -> None:
        scores = score_dataset(
            [
                _submission("hk-a", 20, 10, 10),  # 20
                _submission("hk-b", 60, 30, 30),  # 60
            ]
        )
        assert allocate_dataset_pool(scores, pool=700_000, top_n=10) == {
            "hk-b": 525_000,
            "hk-a": 175_000,
        }

    def test_only_the_top_ten_are_paid(self) -> None:
        scores = score_dataset(
            [_submission(f"hk-{index:02d}", 40, 20 - index, 20 - index) for index in range(12)]
        )
        units = allocate_dataset_pool(scores, pool=700_000, top_n=10)
        assert len(units) == 10
        assert sum(units.values()) == 700_000
        assert "hk-10" not in units
        assert "hk-11" not in units

    def test_a_tie_at_the_cut_goes_to_the_lower_hotkey(self) -> None:
        scores = score_dataset([_submission(f"hk-{index:02d}", 20, 10, 10) for index in range(11)])
        units = allocate_dataset_pool(scores, pool=700_000, top_n=10)
        assert "hk-10" not in units
        assert set(units) == {f"hk-{index:02d}" for index in range(10)}
        assert set(units.values()) == {70_000}

    def test_fewer_than_ten_scorers_divide_the_whole_pool(self) -> None:
        scores = score_dataset([_submission("hk-a", 20, 10, 10), _submission("hk-b", 20, 10, 10)])
        assert allocate_dataset_pool(scores, pool=700_000, top_n=10) == {
            "hk-a": 350_000,
            "hk-b": 350_000,
        }

    def test_zero_scorers_are_never_paid_and_never_take_a_place(self) -> None:
        scores = score_dataset(
            [_submission(f"hk-flood-{index:02d}", 100, 0, 100) for index in range(20)]
            + [_submission("hk-real", 10, 5, 5)]
        )
        assert allocate_dataset_pool(scores, pool=700_000, top_n=10) == {"hk-real": 700_000}

    def test_nobody_scoring_allocates_nothing_so_the_pool_can_burn(self) -> None:
        scores = score_dataset([_submission("hk-a", 0, 0, 0), _submission("hk-b", 10, 0, 10)])
        assert allocate_dataset_pool(scores, pool=700_000, top_n=10) == {}

    def test_allocation_does_not_trust_the_order_it_was_handed(self) -> None:
        scores = [
            DatasetScore("hk-small", 10, 1, 1, Fraction(2)),
            DatasetScore("hk-big", 100, 45, 45, Fraction(90)),
        ]
        assert allocate_dataset_pool(scores, pool=1000, top_n=1) == {"hk-big": 1000}
        assert allocate_dataset_pool(scores[::-1], pool=1000, top_n=1) == {"hk-big": 1000}

    def test_a_top_n_below_one_is_refused(self) -> None:
        with pytest.raises(ScoringError, match="at least 1"):
            allocate_dataset_pool([], pool=100, top_n=0)

    def test_a_negative_pool_is_refused(self) -> None:
        with pytest.raises(ScoringError, match="must not be negative"):
            allocate_dataset_pool([], pool=-1, top_n=10)
