"""Evaluate one miner's model on this validator's own hardware.

This used to post batches to the miner's deployment and score the replies. The
model now arrives as weights: the validator resolves the committed revision,
downloads it, loads it into :class:`~prometheon.evaluation.engine.ModerationEngine`,
and runs the corpus through it locally.

A batch that fails is **scored incorrect, not skipped**: a model that fails to
answer must not come out level with one that answered well, and must not take
the whole cycle down either. An out-of-memory or a checkpoint that will not
load lands there: that batch counts, and counts wrong.

**Running out of time is different.** Each model has one fixed limit,
``[evaluation] model_timeout_seconds``, for loading and judging its whole
corpus. A model that has not finished within it gets no score from this
validator that day, and the items it never reached count for nothing. v3.0
counted them wrong, so the same model scored lower on a slower validator, which
then disagreed with the rest for a reason that had nothing to do with the model
(B28).

One thing did get simpler. There is no longer any question of *whose* model
answered, or whether the thing that answered is the thing that was committed.
The validator resolved the SHA, fetched those bytes, and ran them itself.

Batching is retained even though there is no request to batch into. It bounds
how much work is lost when a model fails partway -- a checkpoint that OOMs on
item 4,000 of 9,000 still scores the 3,999 it answered -- and the clock is
checked between batches.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prometheon.errors import EvaluationError, ValidatorFaultError
from prometheon.evaluation.corpus import LabelledItem, iter_batches
from prometheon.evaluation.engine import ModerationEngine
from prometheon.evaluation.result import MinerEvaluation, MinerEvaluationBuilder

DEFAULT_BATCH_SIZE = 100

#: ``huggingface_hub`` errors that mean the repository itself is gone, private,
#: gated or missing a file: the miner's fault, on every validator alike. Matched
#: by name because the package is an optional extra. Anything else, a full
#: disk, a refused connection, a rate limit, is this validator's problem.
#:
#: Matched by exact name, so a rename is a reclassification. From 1.0 a file
#: missing on the Hub raises ``RemoteEntryNotFoundError``; ``EntryNotFoundError``
#: is only its base class now. Without the new name, a miner's missing file
#: would stop every validator's cycle instead of scoring that one model.
_MINER_SIDE_DOWNLOAD_ERRORS = frozenset(
    {
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "GatedRepoError",
        "DisabledRepoError",
    }
)


class CheckpointUnavailableError(EvaluationError):
    """The committed checkpoint cannot be fetched, through the miner's doing."""

    code = "evaluation.checkpoint_unavailable"


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
    except ImportError as exc:
        # v2 scored this as the miner's failure, so a validator installed as the
        # docs said scored every model zero and still submitted (B10).
        raise ValidatorFaultError(
            "huggingface_hub is not installed, so no model can be downloaded; install the "
            "evaluation runtime with `uv sync --extra wrapper`"
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
        detail = f"{target.hf_repo}@{target.hf_revision}: {type(exc).__name__}: {exc}"
        if type(exc).__name__ in _MINER_SIDE_DOWNLOAD_ERRORS:
            raise CheckpointUnavailableError(f"could not download {detail}") from exc
        raise ValidatorFaultError(
            f"downloading {detail} failed on this validator, not because of the repository; "
            "stopping rather than scoring the model zero"
        ) from exc


def discard_checkpoint(path: str) -> int:
    """Delete a downloaded snapshot, returning the bytes reclaimed.

    The cache is keyed by revision and nothing evicts from it, so a validator
    accumulates every model it has ever scored. That is the right trade on a box
    with room: a miner who re-commits an unchanged model costs one HEAD request,
    and a validator restarting mid-cycle does not re-download the day.

    It is the wrong trade when the disk is smaller than the field. One day's
    eligible set is routinely over 100 GiB of weights, so a host with less than
    that free cannot complete a cycle at all, and the failure arrives as a
    disk-full error partway through scoring rather than as anything nameable.
    Pruning trades bandwidth for space and makes the requirement the size of the
    largest single model instead of the sum of all of them.

    Deletes the revision snapshot and the blobs only it uses. Not the whole repo
    directory, because another revision of the same repo may be a different
    miner's commitment, and **not a blob another revision still links to**:
    revisions share a blob whenever a file is unchanged between them, and v2
    deleted it anyway, leaving the other snapshot pointing at nothing (S9).
    Nothing outside the repo's own ``blobs`` directory is ever deleted, whatever
    a link points at.

    Never raises. Failing to reclaim space is not a reason to stop scoring, and
    the caller has already got what it needed from the path.
    """
    try:
        return _discard_snapshot(Path(path))
    except OSError:
        return 0


def _discard_snapshot(snapshot: Path) -> int:
    import shutil

    target = snapshot.resolve()
    if not target.is_dir():
        return 0
    # `snapshot_download` returns <repo>/snapshots/<sha>, whose files are
    # symlinks into <repo>/blobs. Removing the snapshot alone reclaims nothing.
    repo = target.parent.parent
    blobs_dir = (repo / "blobs").resolve()
    freed = 0
    linked: set[Path] = set()
    for entry in target.rglob("*"):
        # One unreadable entry must not stop the rest being reclaimed.
        try:
            if entry.is_symlink():
                resolved = entry.resolve()
                if resolved.parent == blobs_dir and resolved.is_file():
                    linked.add(resolved)
            elif entry.is_file():
                freed += _size(entry)
        except OSError:
            continue

    still_used = _blobs_linked_from_other_snapshots(repo, target)
    # If the other revisions could not be read, keep every blob: an extra blob
    # costs disk, while deleting a shared one leaves another miner's snapshot
    # pointing at nothing.
    removable = linked - still_used if still_used is not None else set()
    for blob in sorted(removable):
        size = _size(blob)
        try:
            blob.unlink()
        except OSError:
            continue
        freed += size
    shutil.rmtree(target, ignore_errors=True)
    return freed


def _blobs_linked_from_other_snapshots(repo: Path, discarded: Path) -> set[Path] | None:
    """Every blob a sibling revision's snapshot still points at.

    ``None`` when a sibling could not be read completely, which the caller
    treats as "every blob may be in use".
    """
    kept: set[Path] = set()
    snapshots = repo / "snapshots"
    try:
        if not snapshots.is_dir():
            return kept
        for sibling in snapshots.iterdir():
            if sibling.resolve() == discarded:
                continue
            for entry in sibling.rglob("*"):
                if entry.is_symlink():
                    kept.add(entry.resolve())
    except OSError:
        return None
    return kept


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def evaluate_miner(
    *,
    engine: ModerationEngine,
    target: MinerTarget,
    items: Sequence[LabelledItem],
    policy: str,
    policy_version: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    deadline_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> MinerEvaluation:
    """Run ``items`` through ``engine`` and return the result.

    Never raises for a model-side failure. Those are scored, not propagated: a
    checkpoint that will not load, runs out of memory, or produces nothing
    usable is one miner's problem, and every validator hits it simultaneously,
    so an exception escaping here would take down the whole subnet's submission
    at once.

    ``deadline_seconds`` bounds one model, load included, rather than one batch.
    It is checked after the load and after every batch, so a batch already
    running is never cut short. A model still short of its last item when the
    time is up is marked ``timed_out``, and its unfinished items are not
    recorded: it gets no score, so they count neither way.
    """
    if not policy:
        raise EvaluationError("policy text must not be empty")
    if not policy_version:
        raise EvaluationError("policy_version must not be empty")

    builder = MinerEvaluationBuilder(target.hotkey)
    started = clock()

    def out_of_time() -> bool:
        return deadline_seconds is not None and clock() - started > deadline_seconds

    budget = f"{deadline_seconds:.0f}s" if deadline_seconds is not None else ""

    try:
        engine.load()
    except ImportError as exc:
        # Missing torch, transformers or accelerate is this validator's
        # problem, and every model would fail the same way (B10).
        raise ValidatorFaultError(
            f"the evaluation runtime is incomplete ({exc}); install it with "
            "`uv sync --extra wrapper`"
        ) from exc
    except Exception as exc:
        # A checkpoint that will not load scores zero rather than aborting the
        # cycle. Loading reaches arbitrary model code paths inside transformers,
        # so the catch is deliberately wide. Whether the validator itself is
        # healthy is checked separately by the caller, with a canary model,
        # whenever a model fails every batch.
        reason = f"{type(exc).__name__}: {exc}"
        for batch in iter_batches(items, batch_size):
            builder.record_failed_batch(batch, reason=f"model failed to load: {reason}")
        # A load can fail after the weights reached the GPU. Release them, or
        # the next miner's model is the one that runs out of memory.
        try:
            engine.unload()
        except Exception:  # noqa: S110 - best effort; the failure is already recorded
            pass
        return builder.build()

    try:
        done = 0
        if out_of_time():
            builder.record_timeout(
                unfinished=len(items), reason=f"the model took over {budget} to load"
            )
            return builder.build()
        for batch in iter_batches(items, batch_size):
            try:
                result = engine.moderate(policy, [(item.item_id, item.content) for item in batch])
            except Exception as exc:
                builder.record_failed_batch(batch, reason=f"{type(exc).__name__}: {exc}")
            else:
                predicted = {verdict.item_id: verdict.violates for verdict in result.verdicts}
                builder.record_batch(batch, predicted)
            done += len(batch)
            if out_of_time():
                builder.record_timeout(
                    unfinished=len(items) - done,
                    reason=f"the model had judged {done} of {len(items)} items after {budget}",
                )
                break
    finally:
        # One GPU runs every eligible model in turn, so a model left resident is
        # a model the next one cannot fit beside.
        engine.unload()

    return builder.build()


def build_engine(model_path: str, *, dtype: str, device: str, revision: str | None = None) -> Any:
    """An engine for one checkpoint, with numerics pinned from config.

    ``revision`` is threaded through even though ``model_path`` is a snapshot
    already resolved at that commit: an unpinned load is only safe because of
    something that happened earlier, and a pin that is visible at the call site
    stays correct if that stops being true.
    """
    return ModerationEngine(model_path, dtype=dtype, device=device, revision=revision)
