"""Combining the halves: the vector always sums to the pool, burn included."""

from __future__ import annotations

import pytest

from prometheon.config import ScoringConfig
from prometheon.errors import ScoringError
from prometheon.scoring.combine import DEFAULT_WEIGHT_POOL_UNITS, combine_weights
from prometheon.scoring.dataset import DatasetScore, DatasetSubmission, score_dataset
from prometheon.scoring.model import ModelEvaluation, ModelScore, score_models

pytestmark = pytest.mark.unit

DEFAULTS = ScoringConfig()
NO_BURN = ScoringConfig(miner_burn_share_bp=0)  # isolate the allocation math
OWNER = "hk-subnet-owner"


def _dataset(*pairs: tuple[str, int, int]) -> tuple[DatasetScore, ...]:
    return score_dataset(
        DatasetSubmission(hotkey=hotkey, submitted_count=submitted, valid_count=valid)
        for hotkey, submitted, valid in pairs
    )


def _models(*pairs: tuple[str, int]) -> tuple[ModelScore, ...]:
    return score_models(
        [
            ModelEvaluation(
                hotkey=hotkey,
                correct_count=correct,
                scored_count=1000,
                total_tokens=500,
                token_reading_count=1,
            )
            for hotkey, correct in pairs
        ],
        scoring=DEFAULTS,
    ).scores


def test_a_full_field_allocates_the_whole_pool_with_nothing_burned() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 50)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert split.total_units == DEFAULT_WEIGHT_POOL_UNITS
    assert sum(split.weights.values()) == DEFAULT_WEIGHT_POOL_UNITS
    assert split.burn_units == 0
    assert OWNER not in split.weights
    assert dict(split.weights) == {
        "hk-d1": 250_000,
        "hk-d2": 250_000,
        "hk-m1": 300_000,
        "hk-m2": 150_000,
        "hk-m3": 50_000,
    }


def test_the_halves_are_kept_so_a_weight_can_be_explained() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-a", 100, 50)),
        model_scores=_models(("hk-a", 900)),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert dict(split.dataset_units) == {"hk-a": 500_000}
    assert dict(split.model_units) == {"hk-a": 500_000}


def test_a_miner_that_wins_both_halves_is_one_entry_holding_both() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-a", 100, 50), ("hk-b", 100, 50)),
        model_scores=_models(("hk-a", 900), ("hk-c", 800)),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    # Half the dataset pool, plus rank 1 of a two-model field: 3000/4500.
    assert split.weights["hk-a"] == 250_000 + 333_333
    assert split.weights["hk-c"] == 166_667
    assert sum(split.weights.values()) == DEFAULT_WEIGHT_POOL_UNITS
    assert split.burn_units == 0


def test_no_dataset_contributors_burns_the_dataset_half_only() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-a", 40, 0)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 500_000
    assert split.weights[OWNER] == 500_000
    assert "hk-a" not in split.weights
    assert sum(split.weights.values()) == DEFAULT_WEIGHT_POOL_UNITS


def test_no_eligible_models_burns_the_model_half_only() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 50)),
        model_scores=(),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 500_000
    assert dict(split.weights) == {"hk-d1": 250_000, "hk-d2": 250_000, OWNER: 500_000}


def test_an_empty_subnet_burns_everything() -> None:
    split = combine_weights(
        dataset_scores=(),
        model_scores=(),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == DEFAULT_WEIGHT_POOL_UNITS
    assert dict(split.weights) == {OWNER: DEFAULT_WEIGHT_POOL_UNITS}


def test_the_burn_hotkey_may_also_earn_and_its_units_add() -> None:
    split = combine_weights(
        dataset_scores=_dataset((OWNER, 100, 50)),
        model_scores=(),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert split.weights[OWNER] == DEFAULT_WEIGHT_POOL_UNITS
    assert split.burn_units == 500_000
    assert dict(split.dataset_units) == {OWNER: 500_000}


def test_the_share_split_follows_config_and_the_halves_never_drift_apart() -> None:
    scoring = ScoringConfig(dataset_share_bp=7777, miner_burn_share_bp=0)
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50)),
        model_scores=_models(("hk-m1", 900)),
        scoring=scoring,
        burn_hotkey=OWNER,
        total_units=1_000_001,
    )

    # 1_000_001 * 7777 // 10000, floored. The model half takes the whole
    # remainder rather than being computed from its own rounded share.
    assert split.weights["hk-d1"] == 777_700
    assert split.weights["hk-m1"] == 222_301
    assert sum(split.weights.values()) == 1_000_001


def test_the_vector_is_identical_whatever_order_the_snapshot_arrived_in() -> None:
    dataset_pairs = (("hk-d3", 90, 30), ("hk-d1", 100, 50), ("hk-d2", 77, 41))
    model_pairs = (("hk-m2", 811), ("hk-m1", 900), ("hk-m3", 700))

    forward = combine_weights(
        dataset_scores=_dataset(*dataset_pairs),
        model_scores=_models(*model_pairs),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )
    backward = combine_weights(
        dataset_scores=_dataset(*reversed(dataset_pairs)),
        model_scores=_models(*reversed(model_pairs)),
        scoring=NO_BURN,
        burn_hotkey=OWNER,
    )

    assert dict(forward.weights) == dict(backward.weights)
    assert list(forward.weights) == list(backward.weights)


def test_awkward_totals_still_sum_exactly() -> None:
    for total in (1, 7, 999, 65_535, 1_000_003):
        split = combine_weights(
            dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 37)),
            model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
            scoring=NO_BURN,
            burn_hotkey=OWNER,
            total_units=total,
        )
        assert sum(split.weights.values()) == total, total
        assert split.burn_units >= 0


def test_a_non_positive_pool_is_refused() -> None:
    with pytest.raises(ScoringError, match="must be positive"):
        combine_weights(
            dataset_scores=(),
            model_scores=(),
            scoring=NO_BURN,
            burn_hotkey=OWNER,
            total_units=0,
        )


def test_an_empty_burn_hotkey_is_refused_so_units_cannot_vanish() -> None:
    with pytest.raises(ScoringError, match="burn hotkey is required"):
        combine_weights(
            dataset_scores=(),
            model_scores=(),
            scoring=NO_BURN,
            burn_hotkey="",
        )


# --- The fixed burn overlay (miner_burn_share_bp) ---------------------------


def test_the_fixed_burn_share_comes_off_the_top_before_any_miner() -> None:
    # DEFAULTS burns 60% to the owner; miners compete for the other 40%, split
    # 20% dataset / 20% model within that pool.
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 50)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        scoring=DEFAULTS,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 600_000
    assert split.weights[OWNER] == 600_000
    miners = {hk: w for hk, w in split.weights.items() if hk != OWNER}
    assert sum(miners.values()) == 400_000
    assert miners == {
        "hk-d1": 100_000,
        "hk-d2": 100_000,
        "hk-m1": 120_000,
        "hk-m2": 60_000,
        "hk-m3": 20_000,
    }
    assert sum(split.weights.values()) == DEFAULT_WEIGHT_POOL_UNITS


def test_an_empty_half_burns_on_top_of_the_fixed_share() -> None:
    # No models: the model half of the 40% miner pool (20% of total) burns too,
    # so 80% burns and the dataset contributors split the remaining 20%.
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 50)),
        model_scores=(),
        scoring=DEFAULTS,
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 800_000
    assert dict(split.weights) == {"hk-d1": 100_000, "hk-d2": 100_000, OWNER: 800_000}


def test_the_burn_share_is_configurable() -> None:
    split = combine_weights(
        dataset_scores=_dataset(("hk-d1", 100, 50), ("hk-d2", 100, 50)),
        model_scores=_models(("hk-m1", 900), ("hk-m2", 800), ("hk-m3", 700)),
        scoring=ScoringConfig(miner_burn_share_bp=8000),  # 80% burns, 20% to miners
        burn_hotkey=OWNER,
    )

    assert split.burn_units == 800_000
    assert split.weights[OWNER] == 800_000
    assert sum(w for hk, w in split.weights.items() if hk != OWNER) == 200_000
