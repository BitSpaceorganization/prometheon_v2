"""Reclaiming disk after a model has been scored.

The cache is keyed by revision and nothing evicts from it, so a validator
accumulates every model it has ever scored. One day's eligible set is routinely
over 100 GiB of weights, which makes a smaller host unable to finish a cycle at
all: the failure arrives as a disk-full error partway through scoring.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prometheon.evaluation.runner import discard_checkpoint

pytestmark = pytest.mark.unit


def _snapshot(root: Path, *, size: int = 4096) -> Path:
    """A directory shaped like `snapshot_download` leaves one behind.

    The files under `snapshots/<sha>/` are symlinks into `blobs/`. Removing the
    snapshot directory alone reclaims nothing, which is the whole trap here.
    """
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    snap = root / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    blob = blobs / "deadbeef"
    blob.write_bytes(b"x" * size)
    (snap / "model.safetensors").symlink_to(blob)
    (snap / "config.json").write_text("{}")
    return snap


def test_the_blobs_behind_the_symlinks_are_what_gets_reclaimed(tmp_path: Path) -> None:
    snap = _snapshot(tmp_path, size=8192)
    blob = tmp_path / "blobs" / "deadbeef"

    freed = discard_checkpoint(str(snap))

    assert not snap.exists()
    assert not blob.exists(), "deleting the snapshot without the blob frees nothing"
    assert freed >= 8192


def test_a_sibling_revision_of_the_same_repo_survives(tmp_path: Path) -> None:
    """Another revision may be a different miner's commitment."""
    snap = _snapshot(tmp_path)
    other = tmp_path / "snapshots" / ("b" * 40)
    other.mkdir(parents=True)
    keep = tmp_path / "blobs" / "cafe"
    keep.write_bytes(b"y" * 512)
    (other / "model.safetensors").symlink_to(keep)

    discard_checkpoint(str(snap))

    assert other.is_dir()
    assert keep.exists()


def test_a_path_that_is_not_there_is_not_an_error(tmp_path: Path) -> None:
    """Failing to reclaim space must never stop a cycle."""
    assert discard_checkpoint(str(tmp_path / "gone")) == 0
