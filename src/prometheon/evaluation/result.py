"""What one miner's evaluation produced, and how it is accumulated.

The arithmetic lives here rather than in the runner because one rule decides
most of it: **a failed batch scores as all-incorrect, never as skipped.** A
model that returns nothing must not end the cycle with a smaller denominator
and a flattering accuracy, which is what dropping the batch would do. So a
failed batch contributes its items to ``total``, to the confusion matrix in the
cell opposite the truth, and to the per-item record. In that record
``predicted`` is left ``None``, because the model said nothing and the record
must not pretend otherwise.

Token accounting has the mirror-image rule. A failed batch reports no usage, so
its items are outside the mean rather than counted as zero: counting them as
zero would make the least reliable model look like the most efficient one, and
efficiency discounts model score.

Every number that reaches a signed payload is an integer. ``mean_total_tokens``
is a float for scoring's benefit and never leaves this process in that form;
:meth:`MinerEvaluation.to_payload` emits thousandths of a token instead,
because canonical JSON rejects floats outright.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from prometheon.errors import EvaluationError
from prometheon.evaluation.corpus import LabelledItem

#: Failure reasons are operator diagnostics that end up inside a signed,
#: published result. Bounded so a verbose transport error cannot inflate the
#: payload every validator has to sign and store.
MAX_REASON_CHARS: Final[int] = 200

_MILLI: Final[int] = 1000
_BASIS_POINTS: Final[int] = 10000


@dataclass(frozen=True)
class ItemVerdict:
    """One item's outcome, as published for audit."""

    item_id: str
    source: str
    expected_violating: bool
    #: ``None`` when the batch failed and the model produced no answer.
    predicted_violating: bool | None
    correct: bool
    batch_index: int
    batch_failed: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source": self.source,
            "expected": self.expected_violating,
            "predicted": self.predicted_violating,
            "correct": self.correct,
            "batch_index": self.batch_index,
            "batch_failed": self.batch_failed,
        }


@dataclass(frozen=True)
class MinerEvaluation:
    """Everything one miner's evaluation produced, in one immutable record."""

    hotkey: str
    correct: int
    total: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_tokens: int
    #: Items whose batch reported usage: the denominator of the token mean.
    scored_token_items: int
    batches: int
    failed_batches: int
    failed_items: int
    failure_reasons: tuple[str, ...]
    verdicts: tuple[ItemVerdict, ...]

    @property
    def raw_accuracy(self) -> float:
        """Agreement with ground truth over every item, failures included."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total

    @property
    def accuracy_bp(self) -> int:
        """Accuracy in basis points, for payloads that cannot carry floats."""
        if self.total == 0:
            return 0
        return (self.correct * _BASIS_POINTS + self.total // 2) // self.total

    @property
    def has_token_reading(self) -> bool:
        """Whether this model reported usage for a single item.

        ``False`` means every batch failed. Scoring must branch on this rather
        than reading a mean, because there is no honest mean to read.
        """
        return self.scored_token_items > 0

    @property
    def mean_total_tokens(self) -> float | None:
        """Mean prompt+completion tokens per item served, or ``None``.

        Per item rather than per batch: self-exclusion gives miners different
        corpus sizes and different final-batch shapes, and a per-batch mean
        would make that difference look like an efficiency difference.

        ``None`` rather than ``0.0`` when nothing was ever served. Returning
        zero published the least reliable model in the field as the most
        efficient one, the exact inversion this module's contract forbids. The
        intra-miner case was already guarded. The all-batches-failed case was
        not, because there is no non-zero reading left to fall back on.
        """
        if self.scored_token_items == 0:
            return None
        return self.total_tokens / self.scored_token_items

    @property
    def mean_total_tokens_milli(self) -> int | None:
        """:attr:`mean_total_tokens` in thousandths, rounded half up, or ``None``."""
        if self.scored_token_items == 0:
            return None
        return (
            self.total_tokens * _MILLI + self.scored_token_items // 2
        ) // self.scored_token_items

    def to_payload(self, *, include_verdicts: bool = True) -> dict[str, Any]:
        """The signable record. Integers, strings, and booleans only."""
        payload: dict[str, Any] = {
            "hotkey": self.hotkey,
            "correct": self.correct,
            "total": self.total,
            "accuracy_bp": self.accuracy_bp,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total_tokens": self.total_tokens,
            "scored_token_items": self.scored_token_items,
            "mean_total_tokens_milli": self.mean_total_tokens_milli,
            "batches": self.batches,
            "failed_batches": self.failed_batches,
            "failed_items": self.failed_items,
            "failure_reasons": list(self.failure_reasons),
        }
        if include_verdicts:
            payload["verdicts"] = [verdict.to_payload() for verdict in self.verdicts]
        return payload


class MinerEvaluationBuilder:
    """Accumulates batch outcomes into a :class:`MinerEvaluation`."""

    def __init__(self, hotkey: str) -> None:
        if not hotkey:
            raise EvaluationError("an evaluation must be attributed to a hotkey")
        self._hotkey = hotkey
        self._verdicts: list[ItemVerdict] = []
        self._reasons: list[str] = []
        self._correct = 0
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self._total_tokens = 0
        self._scored_token_items = 0
        self._batches = 0
        self._failed_batches = 0
        self._failed_items = 0

    def record_batch(
        self,
        batch: Sequence[LabelledItem],
        predicted: Mapping[str, bool],
        *,
        total_tokens: int,
    ) -> None:
        """Record a batch the model answered completely."""
        if total_tokens < 0:
            raise EvaluationError(f"token usage cannot be negative, got {total_tokens}")

        index = self._batches
        self._batches += 1
        for item in batch:
            if item.item_id not in predicted:
                raise EvaluationError(
                    f"batch {index} was recorded as successful but has no verdict for "
                    f"{item.item_id!r}"
                )
            self._score(item, predicted[item.item_id], batch_index=index, failed=False)

        self._total_tokens += total_tokens
        self._scored_token_items += len(batch)

    def record_failed_batch(self, batch: Sequence[LabelledItem], *, reason: str) -> None:
        """Record a batch that failed: every item counts, and counts wrong."""
        index = self._batches
        self._batches += 1
        self._failed_batches += 1
        self._failed_items += len(batch)
        self._reasons.append(reason[:MAX_REASON_CHARS])
        for item in batch:
            self._score(item, None, batch_index=index, failed=True)

    def build(self) -> MinerEvaluation:
        total = len(self._verdicts)
        # Cheap proof that no path lost or double-counted an item. A confusion
        # matrix that does not sum to the corpus is a scoring bug in here, and
        # it must never reach set_weights.
        assert self._tp + self._fp + self._tn + self._fn == total
        assert self._tp + self._tn == self._correct
        return MinerEvaluation(
            hotkey=self._hotkey,
            correct=self._correct,
            total=total,
            true_positives=self._tp,
            false_positives=self._fp,
            true_negatives=self._tn,
            false_negatives=self._fn,
            total_tokens=self._total_tokens,
            scored_token_items=self._scored_token_items,
            batches=self._batches,
            failed_batches=self._failed_batches,
            failed_items=self._failed_items,
            failure_reasons=tuple(self._reasons),
            verdicts=tuple(self._verdicts),
        )

    def _score(
        self,
        item: LabelledItem,
        predicted: bool | None,
        *,
        batch_index: int,
        failed: bool,
    ) -> None:
        # A failed batch has no answer, so it lands in the cell opposite the
        # truth: an unanswered positive is a miss, an unanswered negative is a
        # false alarm. That keeps the matrix summing to the corpus without
        # inventing a verdict the model never gave.
        effective = (not item.violating) if predicted is None else predicted
        correct = effective == item.violating

        if item.violating:
            if correct:
                self._tp += 1
            else:
                self._fn += 1
        else:
            if correct:
                self._tn += 1
            else:
                self._fp += 1
        if correct:
            self._correct += 1

        self._verdicts.append(
            ItemVerdict(
                item_id=item.item_id,
                source=item.source.value,
                expected_violating=item.violating,
                predicted_violating=predicted,
                correct=correct,
                batch_index=batch_index,
                batch_failed=failed,
            )
        )


__all__ = [
    "MAX_REASON_CHARS",
    "ItemVerdict",
    "MinerEvaluation",
    "MinerEvaluationBuilder",
]
