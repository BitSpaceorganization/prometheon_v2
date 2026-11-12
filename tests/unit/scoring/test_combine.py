"""Combining the pools: shares of the total, a pool burns only when nobody qualifies.

Expected values are worked by hand: 1,000,000 units, 70% data, 25/4/1% models.
"""

from __future__ import annotations

import pytest

from prometheon.errors import ScoringError
from prometheon.rules import RULES, ScoringRules
from prometheon.scoring.combine import DEFAULT_WEIGHT_POOL_UNITS, combine_weights
from prometheon.scoring.dataset import DatasetScore, DatasetSubmission, score_dataset
from prometheon.scoring.model import ModelEvaluation, ModelScore, score_models

pytestmark = pytest.mark.unit

# Stated rather than inherited, so these tests exercise the arithmetic of the
# split they name. The shipped split is pinned in
# `tests/contract/test_shipped_artifacts.py`.
V3 = ScoringRules(dataset_share_bp=7000, dataset_top_n=10, model_rank_shares_bp=(2500, 400, 100))
OWNER = "hk-subnet-owner"


def _dataset(*rows: tuple[str, int, int, int]) -> tuple[DatasetScore, ...]:
    return score_dataset(
        DatasetSubmission(
            hotkey=hotkey,
            submitted_count=submitted,
            violating_confirmed=v_pos,
            compliant_confirmed=v_neg,
        )
        for hotkey, submitted, v_pos, v_neg in rows
    )


def _models(*pairs: tuple[str, int]) -> tuple[ModelScore, ...]:
    """Models under 500M parameters, so no penalty, each a whole point apart or more."""
    return score_models(
        [
            ModelEvaluation(
                hotkey=hotkey,
                correct_count=correct,
                scored_count=1000,
                commit_block=100 + position,
                parameters=400_000_000,
            )
            for position, (hotkey, correct) in enumerate(pairs)
        ],
        rules=V3,
    ).scores


def test_a_full_field_pays_seventy_twenty_five_four_one_and_burns_nothing() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 20, 10, 10), ("hk-d2", 20, 10, 10)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert dict(split.weights) == {
        "hk-d1": 350_000,
        "hk-d2": 350_000,
        "hk-m1": 250_000,
        "hk-m2": 40_000,
        "hk-m3": 10_000,
    }
    assert split.burn_units == 0
    assert OWNER not in split.weights
    assert sum(split.weights.values()) == DEFAULT_WEIGHT_POOL_UNITS


def test_one_model_collects_the_whole_thirty_percent() -> None:
    """An unfilled place goes to the lowest winner present, so nothing burns."""
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 20, 10, 10)),
        model_scores=_models(("hk-m1", 600)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert dict(split.weights) == {"hk-d1": 700_000, "hk-m1": 300_000}
    assert split.burn_units == 0


def test_two_models_take_twenty_five_and_five_percent() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 20, 10, 10)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert split.weights["hk-m1"] == 250_000
    assert split.weights["hk-m2"] == 50_000
    assert split.burn_units == 0


def test_a_fourth_model_is_not_paid() -> None:
    split = combine_weights(
        dataset_scores=(),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700), ("hk-m4", 600)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert "hk-m4" not in split.weights
    assert dict(split.model_units) == {"hk-m1": 250_000, "hk-m2": 40_000, "hk-m3": 10_000}


def test_no_data_contributor_burns_the_seventy_percent() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-flood", 500, 0, 500)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 700_000
    assert "hk-flood" not in split.weights


def test_no_model_burns_the_thirty_percent() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 20, 10, 10)),
        model_scores=(),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert dict(split.weights) == {"hk-d1": 700_000, OWNER: 300_000}


def test_an_empty_subnet_burns_everything() -> None:
    split = combine_weights(dataset_scores=(), model_scores=(), rules=V3, burn_hotkey=OWNER)

    assert dict(split.weights) == {OWNER: DEFAULT_WEIGHT_POOL_UNITS}
    assert split.burn_units == DEFAULT_WEIGHT_POOL_UNITS
    assert not split.earned_units


def test_a_miner_in_both_pools_is_one_entry_holding_both() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-a", 20, 10, 10), ("hk-b", 20, 10, 10)),
        model_scores=_models(("hk-a", 900)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    # A is the only model, so it holds the whole 30%.
    assert split.weights["hk-a"] == 350_000 + 300_000
    assert split.earned_units["hk-a"] == 650_000
    assert split.burn_units == 0


def test_the_burn_hotkey_may_also_earn_and_its_earnings_stay_separate_from_burn() -> None:
    """B5: a published record must be able to tell the owner's earnings from burn."""
    split = combine_weights(
        dataset_scores=_dataset((OWNER, 20, 10, 10)),
        model_scores=(),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert split.weights[OWNER] == DEFAULT_WEIGHT_POOL_UNITS
    assert split.earned_units == {OWNER: 700_000}
    assert split.burn_units == 300_000


def test_shares_below_one_hundred_percent_leave_a_standing_burn() -> None:
    rules = ScoringRules(dataset_share_bp=6000, model_rank_shares_bp=(2500, 400, 100))
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 20, 10, 10)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        rules=rules,
        burn_hotkey=OWNER,
    )

    assert split.weights["hk-d1"] == 600_000
    assert split.burn_units == 100_000


def test_the_vector_is_identical_whatever_order_the_inputs_arrived_in() -> None:
    dataset_rows = (("hk-d3", 90, 30, 20), ("hk-d1", 100, 50, 50), ("hk-d2", 77, 21, 20))
    model_pairs = (("hk-m2", 811), ("hk-m1", 900), ("hk-m3", 700))

    forward = combine_weights(
        dataset_scores=_dataset(*dataset_rows),
        model_scores=_models(*model_pairs),
        rules=V3,
        burn_hotkey=OWNER,
    )
    backward = combine_weights(
        dataset_scores=_dataset(*reversed(dataset_rows)),
        model_scores=_models(*reversed(model_pairs)),
        rules=V3,
        burn_hotkey=OWNER,
    )

    assert dict(forward.weights) == dict(backward.weights)


@pytest.mark.parametrize("total", [1, 7, 999, 65_535, 1_000_003])
def test_awkward_totals_still_sum_exactly(total: int) -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50, 50), ("hk-d2", 100, 37, 30)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        rules=V3,
        burn_hotkey=OWNER,
        total_units=total,
    )
    assert sum(split.weights.values()) == total
    assert split.burn_units >= 0


def test_a_non_positive_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="must be positive"):
        combine_weights(
            dataset_scores=(), model_scores=(), rules=V3, burn_hotkey=OWNER, total_units=0
        )


def test_an_empty_burn_hotkey_is_refused_so_units_cannot_vanish() -> None:
    with pytest.raises(ScoringError, match="burn hotkey is required"):
        combine_weights(dataset_scores=(), model_scores=(), rules=V3, burn_hotkey="")


class TestTheSharesAreValidatedWhenTheRulesAreMade:
    def test_shares_over_one_hundred_percent_are_refused(self) -> None:
        with pytest.raises(ScoringError, match="over 100%"):
            ScoringRules(dataset_share_bp=8000, model_rank_shares_bp=(2500, 400, 100))

    def test_the_published_rules_are_the_v3_split(self) -> None:
        assert RULES.dataset_share_bp == 7000
        assert RULES.dataset_top_n == 10
        assert RULES.model_rank_shares_bp == (2500, 400, 100)
