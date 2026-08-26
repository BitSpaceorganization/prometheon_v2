"""``prometheon model commit`` as a reconciler.

The flag exists because of one asymmetry: writing a commitment that is already
on chain is not a no-op. It resets the block that
:func:`~prometheon.registry.validation._duplicate_losers` sorts on, so an
unchanged re-commit moves the miner *behind* anyone holding an earlier copy of
the same revision SHA. A loop that re-commits on a timer therefore hands a
copycat a fresh advantage on every tick, which is the opposite of what the
operator running it believes it is doing.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import pytest

from prometheon.chain.commitment import ModelCommitment
from prometheon.cli import miner
from prometheon.errors import CommitmentError

pytestmark = pytest.mark.unit

REPO = "acme/guard"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40


class _Recorder:
    """Stands in for every chain call ``cmd_commit`` makes."""

    def __init__(self, *, current: Any) -> None:
        self.current = current
        self.published: list[ModelCommitment] = []

    def read_commitment(self, subtensor: Any, **kwargs: Any) -> ModelCommitment | None:
        if isinstance(self.current, Exception):
            raise self.current
        return self.current

    def publish_commitment(self, subtensor: Any, **kwargs: Any) -> str:
        self.published.append(kwargs["commitment"])
        return "0xreceipt"


@pytest.fixture
def commit(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run ``cmd_commit`` against a fake chain and report what it wrote."""

    def run(*, current: Any, if_changed: bool, revision: str = REVISION) -> _Recorder:
        recorder = _Recorder(current=current)
        config = SimpleNamespace(
            chain=SimpleNamespace(netuid=108, network="finney"),
            miner=SimpleNamespace(hf_repo=REPO, hf_revision=revision),
        )
        wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5Alice"))

        monkeypatch.setattr(miner, "load", lambda _path: config)
        monkeypatch.setattr(miner, "open_wallet", lambda _config: wallet)
        monkeypatch.setattr(miner.chain, "connect", lambda _network: object())
        monkeypatch.setattr(
            miner.chain,
            "sync_metagraph_view",
            lambda *a, **k: SimpleNamespace(uid_for=lambda _h: 7),
        )
        monkeypatch.setattr(miner, "read_commitment", recorder.read_commitment)
        monkeypatch.setattr(miner, "publish_commitment", recorder.publish_commitment)

        args = argparse.Namespace(
            config=None, hf_repo=None, hf_revision=None, dry_run=False, if_changed=if_changed
        )
        assert miner.cmd_commit(args) == 0
        return recorder

    return run


class TestCommitOnlyIfChanged:
    def test_an_identical_commitment_is_not_rewritten(self, commit: Any) -> None:
        """The regression this exists for.

        Rewriting it would re-date the claim and cost the miner every duplicate
        contest against an earlier mirror of the same revision.
        """
        recorder = commit(
            current=ModelCommitment(hf_repo=REPO, hf_revision=REVISION), if_changed=True
        )

        assert recorder.published == []

    def test_a_changed_revision_is_written(self, commit: Any) -> None:
        recorder = commit(
            current=ModelCommitment(hf_repo=REPO, hf_revision=OTHER_REVISION), if_changed=True
        )

        assert [c.hf_revision for c in recorder.published] == [REVISION]

    def test_a_changed_repo_at_the_same_revision_is_written(self, commit: Any) -> None:
        """A mirror of your own model under another name is still a change."""
        recorder = commit(
            current=ModelCommitment(hf_repo="someone-else/guard", hf_revision=REVISION),
            if_changed=True,
        )

        assert [c.hf_repo for c in recorder.published] == [REPO]

    def test_a_missing_commitment_is_written(self, commit: Any) -> None:
        recorder = commit(current=None, if_changed=True)

        assert len(recorder.published) == 1

    def test_an_unreadable_commitment_is_repaired_rather_than_skipped(self, commit: Any) -> None:
        """Only an exact match is a reason to leave the chain alone.

        A decode failure means the chain holds something this miner cannot
        confirm is theirs, and the loop's job is to fix that.
        """
        recorder = commit(current=CommitmentError("undecodable"), if_changed=True)

        assert len(recorder.published) == 1

    def test_without_the_flag_an_identical_commitment_is_still_rewritten(self, commit: Any) -> None:
        """The default is unchanged: an explicit `commit` writes."""
        recorder = commit(
            current=ModelCommitment(hf_repo=REPO, hf_revision=REVISION), if_changed=False
        )

        assert len(recorder.published) == 1

    def test_the_chain_is_not_even_read_without_the_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No extra round-trip for callers that did not ask for one."""
        reads: list[Any] = []
        recorder = _Recorder(current=None)
        config = SimpleNamespace(
            chain=SimpleNamespace(netuid=108, network="finney"),
            miner=SimpleNamespace(hf_repo=REPO, hf_revision=REVISION),
        )
        monkeypatch.setattr(miner, "load", lambda _path: config)
        monkeypatch.setattr(
            miner,
            "open_wallet",
            lambda _c: SimpleNamespace(hotkey=SimpleNamespace(ss58_address="5A")),
        )
        monkeypatch.setattr(miner.chain, "connect", lambda _n: object())
        monkeypatch.setattr(miner, "publish_commitment", recorder.publish_commitment)
        monkeypatch.setattr(miner, "read_commitment", lambda *a, **k: reads.append(1))

        miner.cmd_commit(
            argparse.Namespace(
                config=None, hf_repo=None, hf_revision=None, dry_run=False, if_changed=False
            )
        )

        assert reads == []
