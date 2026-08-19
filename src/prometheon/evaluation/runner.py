"""Evaluate one miner's model on this validator's own hardware.

This used to post batches to the miner's deployment and score the replies. The
model now arrives as weights: the validator resolves the committed revision,
downloads it, loads it into :class:`~prometheon.evaluation.engine.ModerationEngine`,
and runs the corpus through it locally.

The scoring contract is unchanged, and deliberately so. A batch that cannot be
completed is **scored incorrect, not skipped** -- a model that fails to answer
must not come out level with one that answered well, and must not take the
whole cycle down either. What used to fail as a timeout or a malformed response
now fails as an out-of-memory or a checkpoint that will not load, and both land
in the same place: that batch counts, and counts wrong.

One thing did get simpler. There is no longer any question of *whose* model
answered, or whether the thing that answered is the thing that was committed.
The validator resolved the SHA, fetched those bytes, and ran them itself.

Batching is retained even though there is no request to batch into. It bounds
how much work is lost when a model fails partway -- a checkpoint that OOMs on
item 4,000 of 9,000 still scores the 3,999 it answered -- and it keeps the
token accounting per batch, which is what feeds the efficiency term.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prometheon.errors import EvaluationError
from prometheon.evaluation.corpus import LabelledItem, iter_batches
from prometheon.evaluation.engine import ModerationEngine
from prometheon.evaluation.result import MinerEvaluation, MinerEvaluationBuilder

DEFAULT_BATCH_SIZE = 100


@dataclass(frozen=True)
class MinerTarget:
    """Which checkpoint to run for one miner."""

    hotkey: str
    hf_repo: str
    hf_revision: str

    def __post_init__(self) -> None:
        for name, value in (
            ("hotkey", self.hotkey),
            ("hf_repo", self.hf_repo),
            ("hf_revision", self.hf_revision),
        ):
            if not value:
                raise EvaluationError(f"{name} must not be empty")


def download_checkpoint(target: MinerTarget, *, allow_patterns: Sequence[str] | None = None) -> str:
    """Fetch the committed revision and return the local path.

    Pinned to the exact SHA, never a branch: the commitment names a commit, and
    resolving anything else would score a different model than the one the
    miner froze on chain.

    Cached by revision, so a miner who re-commits an unchanged model costs one
    HEAD request rather than a second download, and a validator restarting
    mid-cycle does not start the day's bandwidth again.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on the wrapper extra
        raise EvaluationError(
            "huggingface_hub is required to evaluate models locally; install the 'wrapper' extra"
        ) from exc

    try:
        return str(
            snapshot_download(
                target.hf_repo,
                revision=target.hf_revision,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
            )
        )
    except Exception as exc:
        raise EvaluationError(
            f"could not download {target.hf_repo}@{target.hf_revision}: {type(exc).__name__}: {exc}"
        ) from exc


def evaluate_miner(
    *,
    engine: ModerationEngine,
    target: MinerTarget,
    items: Sequence[LabelledItem],
    policy: str,
    policy_version: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    deadline_seconds: float | None = None,
) -> MinerEvaluation:
    """Run ``items`` through ``engine`` and return the result.

    Never raises for a model-side failure. Those are scored, not propagated: a
    checkpoint that will not load, runs out of memory, or produces nothing
    usable is one miner's problem, and every validator hits it simultaneously,
    so an exception escaping here would take down the whole subnet's submission
    at once.

    ``deadline_seconds`` bounds one model rather than one batch. A model slow
    enough to threaten the cycle is scored on what it completed, and the
    remainder counts against it -- being unable to finish the corpus is a
    property of the model, not an accident to be forgiven.
    """
    if not policy:
        raise EvaluationError("policy text must not be empty")
    if not policy_version:
        raise EvaluationError("policy_version must not be empty")

    builder = MinerEvaluationBuilder(target.hotkey)
    started = time.monotonic()

    try:
        engine.load()
    except Exception as exc:
        # A checkpoint that will not load scores zero rather than aborting the
        # cycle. Loading reaches arbitrary model code paths inside transformers,
        # so the catch is deliberately wide: `EngineError` covers the cases the
        # engine recognises, and everything else still has to be one miner's
        # failure rather than the subnet's.
        reason = f"{type(exc).__name__}: {exc}"
        for batch in iter_batches(items, batch_size):
            builder.record_failed_batch(batch, reason=f"model failed to load: {reason}")
        return builder.build()

    try:
        for batch in iter_batches(items, batch_size):
            if deadline_seconds is not None and time.monotonic() - started > deadline_seconds:
                builder.record_failed_batch(
                    batch,
                    reason=(f"model exceeded its {deadline_seconds:.0f}s budget before this batch"),
                )
                continue
            try:
                result = engine.moderate(policy, [(item.item_id, item.content) for item in batch])
            except Exception as exc:
                builder.record_failed_batch(batch, reason=f"{type(exc).__name__}: {exc}")
            else:
                predicted = {verdict.item_id: verdict.violates for verdict in result.verdicts}
                builder.record_batch(
                    batch,
                    predicted,
                    total_tokens=result.prompt_tokens + result.completion_tokens,
                )
    finally:
        # One GPU runs every eligible model in turn, so a model left resident is
        # a model the next one cannot fit beside.
        engine.unload()

    return builder.build()


def build_engine(model_path: str, *, dtype: str, device: str) -> Any:
    """An engine for one checkpoint, with numerics pinned from config."""
    return ModerationEngine(model_path, dtype=dtype, device=device)
