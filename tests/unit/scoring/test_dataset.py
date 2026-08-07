"""Dataset contribution: V^2/S, the top-N cut, and the burn case."""

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


def _submission(hotkey: str, submitted: int, valid: int) -> DatasetSubmission:
    return DatasetSubmission(hotkey=hotkey, submitted_count=submitted, valid_count=valid)


def test_the_worked_example_one_hundred_submitted_fifty_valid_scores_twenty_five() -> None:
    (score,) = score_dataset([_submission("hk-a", 100, 50)])

    assert score.score == Fraction(25)


def test_precision_beats_volume_at_equal_valid_counts() -> None:
    scores = score_dataset(
        [
            _submission("hk-flooder", 1000, 50),
            _submission("hk-careful", 60, 50),
        ]
    )

    assert [entry.hotkey for entry in scores] == ["hk-careful", "hk-flooder"]
    assert scores[0].score == Fraction(2500, 60)
    assert scores[1].score == Fraction(2500, 1000)


def test_no_submissions_scores_zero_without_dividing_by_zero() -> None:
    (score,) = score_dataset([_submission("hk-a", 0, 0)])

    assert score.score == Fraction(0)


def test_a_perfect_contributor_scores_its_valid_count() -> None:
    # V == S collapses V * (V/S) to V, which is the shape the design intends.
    (score,) = score_dataset([_submission("hk-a", 40, 40)])

    assert score.score == Fraction(40)


def test_equal_scores_rank_by_hotkey_ascending() -> None:
    scores = score_dataset(
        [
            _submission("hk-z", 100, 50),
            _submission("hk-a", 100, 50),
            _submission("hk-m", 100, 50),
        ]
    )

    assert [entry.hotkey for entry in scores] == ["hk-a", "hk-m", "hk-z"]


def test_more_valid_than_submitted_is_refused_as_an_inconsistent_snapshot() -> None:
    with pytest.raises(ScoringError, match="more valid items than submitted"):
        score_dataset([_submission("hk-a", 10, 11)])


def test_negative_counts_are_refused() -> None:
    with pytest.raises(ScoringError, match="negative dataset counts"):
        score_dataset([_submission("hk-a", -1, 0)])


def test_a_repeated_hotkey_is_refused_rather_than_scored_twice() -> None:
    with pytest.raises(ScoringError, match="duplicate"):
        score_dataset([_submission("hk-a", 10, 5), _submission("hk-a", 10, 5)])


def test_the_pool_splits_in_proportion_to_score() -> None:
    scores = score_dataset(
        [
            _submission("hk-a", 100, 50),  # 25
            _submission("hk-b", 100, 87),  # 75.69
        ]
    )

    units = allocate_dataset_pool(scores, pool=1000, top_n=10)

    assert sum(units.values()) == 1000
    assert units["hk-b"] > units["hk-a"]
    assert units == {"hk-b": 752, "hk-a": 248}


def test_only_the_top_n_are_paid_and_the_rest_of_the_pool_still_goes_to_them() -> None:
    scores = score_dataset(
        [_submission(f"hk-{index:02d}", 100, 100 - index) for index in range(12)]
    )

    units = allocate_dataset_pool(scores, pool=500_000, top_n=10)

    assert len(units) == 10
    assert sum(units.values()) == 500_000
    # The two weakest contributors are cut, not paid a smaller share.
    assert "hk-10" not in units
    assert "hk-11" not in units


def test_a_field_smaller_than_top_n_divides_the_whole_pool_among_those_present() -> None:
    scores = score_dataset([_submission("hk-a", 100, 50), _submission("hk-b", 100, 50)])

    units = allocate_dataset_pool(scores, pool=500_000, top_n=10)

    assert units == {"hk-a": 250_000, "hk-b": 250_000}


def test_a_single_contributor_takes_the_entire_pool() -> None:
    scores = score_dataset([_submission("hk-a", 3, 1)])

    assert allocate_dataset_pool(scores, pool=500_000, top_n=10) == {"hk-a": 500_000}


def test_all_zero_scores_allocate_nothing_so_the_pool_can_burn() -> None:
    scores = score_dataset([_submission("hk-a", 0, 0), _submission("hk-b", 10, 0)])

    assert allocate_dataset_pool(scores, pool=500_000, top_n=10) == {}


def test_no_submissions_at_all_allocate_nothing() -> None:
    assert allocate_dataset_pool(score_dataset([]), pool=500_000, top_n=10) == {}


def test_zero_scorers_never_displace_a_real_contributor_from_the_cut() -> None:
    scores = score_dataset(
        [_submission(f"hk-zero-{index}", 10, 0) for index in range(20)]
        + [_submission("hk-real", 10, 5)]
    )

    assert allocate_dataset_pool(scores, pool=1000, top_n=10) == {"hk-real": 1000}


def test_allocation_does_not_trust_the_order_it_was_handed() -> None:
    scores = [
        DatasetScore(hotkey="hk-small", submitted_count=100, valid_count=10, score=Fraction(1)),
        DatasetScore(hotkey="hk-big", submitted_count=100, valid_count=90, score=Fraction(81)),
    ]

    ranked_first = allocate_dataset_pool(list(reversed(scores)), pool=1000, top_n=1)
    as_given = allocate_dataset_pool(scores, pool=1000, top_n=1)

    assert as_given == ranked_first == {"hk-big": 1000}


def test_a_top_n_below_one_is_refused() -> None:
    with pytest.raises(ScoringError, match="at least 1"):
        allocate_dataset_pool([], pool=100, top_n=0)


def test_a_negative_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="must not be negative"):
        allocate_dataset_pool([], pool=-1, top_n=10)
