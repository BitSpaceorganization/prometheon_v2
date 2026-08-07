"""Accumulation arithmetic and the shape of the published record."""

from __future__ import annotations

from typing import Any

import pytest

from prometheon.errors import CanonicalEncodingError, EvaluationError
from prometheon.evaluation.corpus import ItemSource, LabelledItem
from prometheon.evaluation.result import (
    MAX_REASON_CHARS,
    MinerEvaluation,
    MinerEvaluationBuilder,
)
from prometheon.security.canonical_json import DOMAIN_EVALUATION, domain_prefixed_bytes

pytestmark = pytest.mark.unit

HOTKEY = "5FA0miner_a"


def item(item_id: str, *, violating: bool) -> LabelledItem:
    return LabelledItem(
        item_id=item_id,
        content=f"content {item_id}",
        violating=violating,
        source=ItemSource.PRODUCTION,
    )


def batch(positives: int, negatives: int, prefix: str = "i") -> list[LabelledItem]:
    return [item(f"{prefix}p{n}", violating=True) for n in range(positives)] + [
        item(f"{prefix}n{n}", violating=False) for n in range(negatives)
    ]


def all_correct(items: list[LabelledItem]) -> dict[str, bool]:
    return {entry.item_id: entry.violating for entry in items}


# --- the confusion matrix ---------------------------------------------------


def test_every_cell_is_filled_from_the_labels_and_the_answers() -> None:
    items = batch(2, 2)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(
        items,
        {
            "ip0": True,  # true positive
            "ip1": False,  # false negative
            "in0": False,  # true negative
            "in1": True,  # false positive
        },
        total_tokens=44,
    )
    result = builder.build()

    assert (result.true_positives, result.false_negatives) == (1, 1)
    assert (result.true_negatives, result.false_positives) == (1, 1)
    assert result.correct == 2
    assert result.total == 4


def test_the_matrix_always_sums_to_the_corpus() -> None:
    items = batch(30, 70)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(items, all_correct(items), total_tokens=1100)
    builder.record_failed_batch(batch(10, 10, prefix="j"), reason="unreachable")
    result = builder.build()

    cells = (
        result.true_positives,
        result.false_positives,
        result.true_negatives,
        result.false_negatives,
    )
    assert sum(cells) == result.total == 120
    assert result.true_positives + result.true_negatives == result.correct == 100


# --- failed batches ---------------------------------------------------------


def test_a_failed_batch_counts_every_item_as_incorrect() -> None:
    items = batch(4, 6)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_failed_batch(items, reason="HTTP 503")
    result = builder.build()

    assert result.total == 10
    assert result.correct == 0
    assert result.failed_batches == 1
    assert result.failed_items == 10
    assert result.false_negatives == 4, "unanswered positives are misses"
    assert result.false_positives == 6, "unanswered negatives are false alarms"
    assert result.raw_accuracy == 0.0


def test_a_failed_batch_reports_no_verdict_rather_than_a_guess() -> None:
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_failed_batch(batch(1, 1), reason="timeout")
    result = builder.build()

    assert all(verdict.predicted_violating is None for verdict in result.verdicts)
    assert all(verdict.batch_failed for verdict in result.verdicts)


def test_a_failed_batch_does_not_dilute_the_token_mean() -> None:
    good = batch(5, 5)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(good, all_correct(good), total_tokens=200)
    builder.record_failed_batch(batch(5, 5, prefix="j"), reason="unreachable")
    result = builder.build()

    assert result.scored_token_items == 10
    assert result.mean_total_tokens == pytest.approx(20.0)
    assert result.total == 20


def test_batch_indexes_are_sequential_across_successes_and_failures() -> None:
    first = batch(1, 1, prefix="a")
    third = batch(1, 1, prefix="c")
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(first, all_correct(first), total_tokens=10)
    builder.record_failed_batch(batch(1, 1, prefix="b"), reason="boom")
    builder.record_batch(third, all_correct(third), total_tokens=10)
    result = builder.build()

    assert result.batches == 3
    assert [verdict.batch_index for verdict in result.verdicts] == [0, 0, 1, 1, 2, 2]


def test_failure_reasons_are_truncated() -> None:
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_failed_batch(batch(1, 0), reason="x" * 5000)
    result = builder.build()

    assert len(result.failure_reasons[0]) == MAX_REASON_CHARS


# --- derived values ---------------------------------------------------------


def test_an_empty_evaluation_divides_by_nothing() -> None:
    result = MinerEvaluationBuilder(HOTKEY).build()

    assert result.total == 0
    assert result.raw_accuracy == 0.0
    assert result.accuracy_bp == 0
    # None rather than 0.0, because "never measured" is not "free". Publishing
    # zero here made a model that served nothing the most efficient in the
    # field.
    assert result.mean_total_tokens is None
    assert result.mean_total_tokens_milli is None
    assert result.has_token_reading is False


def test_accuracy_basis_points_round_half_up() -> None:
    items = batch(1, 2)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(items, {"ip0": True, "in0": False, "in1": True}, total_tokens=3)
    result = builder.build()

    assert result.raw_accuracy == pytest.approx(2 / 3)
    assert result.accuracy_bp == 6667


def test_the_token_mean_keeps_three_decimal_places_as_an_integer() -> None:
    items = batch(1, 2)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(items, all_correct(items), total_tokens=10)
    result = builder.build()

    assert result.mean_total_tokens == pytest.approx(10 / 3)
    assert result.mean_total_tokens_milli == 3333


# --- builder invariants -----------------------------------------------------


def test_a_missing_verdict_is_a_programming_error_not_a_silent_zero() -> None:
    items = batch(2, 0)
    builder = MinerEvaluationBuilder(HOTKEY)
    with pytest.raises(EvaluationError, match="no verdict for"):
        builder.record_batch(items, {"ip0": True}, total_tokens=10)


def test_negative_usage_is_refused() -> None:
    items = batch(1, 0)
    builder = MinerEvaluationBuilder(HOTKEY)
    with pytest.raises(EvaluationError, match="negative"):
        builder.record_batch(items, all_correct(items), total_tokens=-1)


def test_an_evaluation_must_name_a_hotkey() -> None:
    with pytest.raises(EvaluationError, match="hotkey"):
        MinerEvaluationBuilder("")


# --- the published record ---------------------------------------------------


def built() -> MinerEvaluation:
    good = batch(3, 3)
    builder = MinerEvaluationBuilder(HOTKEY)
    builder.record_batch(good, all_correct(good), total_tokens=70)
    builder.record_failed_batch(batch(1, 1, prefix="j"), reason="HTTP 500")
    return builder.build()


def walk(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for item_value in value.values() for leaf in walk(item_value)]
    if isinstance(value, list):
        return [leaf for item_value in value for leaf in walk(item_value)]
    return [value]


def test_the_payload_carries_no_floats_and_can_be_signed() -> None:
    """Canonical JSON rejects floats, so the record must not contain one."""
    payload = built().to_payload()

    assert not any(isinstance(leaf, float) for leaf in walk(payload))
    assert domain_prefixed_bytes(DOMAIN_EVALUATION, payload)


def test_a_float_in_the_record_would_be_caught_by_the_encoder() -> None:
    """Guards the test above: prove the encoder is what is doing the work."""
    payload = built().to_payload()
    payload["mean_total_tokens"] = 11.5
    with pytest.raises(CanonicalEncodingError):
        domain_prefixed_bytes(DOMAIN_EVALUATION, payload)


def test_the_payload_reports_the_counts_a_dashboard_needs() -> None:
    payload = built().to_payload()

    assert payload["hotkey"] == HOTKEY
    assert payload["total"] == 8
    assert payload["correct"] == 6
    assert payload["failed_batches"] == 1
    assert payload["failed_items"] == 2
    assert payload["batches"] == 2
    assert payload["mean_total_tokens_milli"] == 11667
    assert payload["accuracy_bp"] == 7500
    assert len(payload["verdicts"]) == 8


def test_verdicts_can_be_left_out_of_the_payload() -> None:
    payload = built().to_payload(include_verdicts=False)

    assert "verdicts" not in payload
    assert payload["total"] == 8


def test_a_verdict_record_names_its_item_and_its_source() -> None:
    record = built().verdicts[0].to_payload()

    assert set(record) == {
        "item_id",
        "source",
        "expected",
        "predicted",
        "correct",
        "batch_index",
        "batch_failed",
    }
    assert record["source"] == "production"
