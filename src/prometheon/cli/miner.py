"""Miner commands: render the wrapper, commit it, and check it before a cycle does.

The three exist as separate steps because they fail for different reasons and at
different costs. Rendering is local and free. Committing writes to the chain and
spends a slice of a per-epoch byte quota. Verification calls Hugging Face and
Hugging Face, and it is the only one that can tell a miner what a *validator* will
conclude, which is the answer that decides whether they earn anything.

``verify`` runs the validator's own
:class:`~prometheon.registry.validation.ModelRegistry` rather than a
reimplementation of it. A separate "miner-side check" would be a second
definition of eligibility that could agree with the first for months and then
diverge on the day it mattered.
"""

from __future__ import annotations

import argparse

from prometheon.chain import subtensor as chain
from prometheon.chain.commitment import (
    COMMITMENT_MAX_BYTES,
    ModelCommitment,
    commitment_size_bytes,
    encode_commitment,
    publish_commitment,
    read_commitment,
)
from prometheon.cli._common import EXIT_FAILED, EXIT_OK, load, note, open_wallet, out, table
from prometheon.errors import ConfigError
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.validation import MinerEntry, ModelRegistry
from prometheon.revision import require_valid_revision


def _commitment_from(args: argparse.Namespace) -> ModelCommitment:
    """Build the commitment from flags, falling back to ``[miner]`` in config.

    Flags win over config so a miner can deploy a new revision without editing
    a file, but the config exists so the common case is one short command.
    """
    config = load(args.config)
    hf_repo = args.hf_repo or config.miner.hf_repo
    hf_revision = args.hf_revision or config.miner.hf_revision

    missing = [
        name
        for name, value in (
            ("--hf-repo", hf_repo),
            ("--hf-revision", hf_revision),
        )
        if not value
    ]
    if missing:
        raise ConfigError(
            f"missing {', '.join(missing)}; pass them as flags or set them under "
            "[miner] in the config file"
        )

    # Checked here rather than at the chain call: a miner who typed a branch
    # name should learn that before an extrinsic spends part of their quota.
    require_valid_revision(hf_revision)
    return ModelCommitment(hf_repo=hf_repo, hf_revision=hf_revision)


def cmd_commit(args: argparse.Namespace) -> int:
    """Write ``(hf_repo, hf_revision)`` on chain. This is the submission."""
    config = load(args.config)
    commitment = _commitment_from(args)
    payload = encode_commitment(commitment)
    size = commitment_size_bytes(commitment)

    out(
        table(
            [
                ("hf_repo", commitment.hf_repo),
                ("hf_revision", commitment.hf_revision),
                ("payload", payload),
                ("size", f"{size}/{COMMITMENT_MAX_BYTES} bytes"),
            ]
        )
    )

    if args.dry_run:
        note("")
        note("dry run: nothing was written to the chain")
        return EXIT_OK

    wallet = open_wallet(config)
    subtensor = chain.connect(config.chain.network)
    written = publish_commitment(
        subtensor, wallet=wallet, netuid=config.chain.netuid, commitment=commitment
    )
    note("")
    note(f"committed on netuid={config.chain.netuid}: {written}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the validator's own eligibility pipeline against your deployment.

    Reads the commitment from the chain rather than from flags, so what is
    checked is what a validator will actually read. A miner who has not
    committed yet can pass ``--hf-repo``/``--hf-revision`` to dry
    run the checks first.
    """
    config = load(args.config)

    if args.hf_revision:
        commitment = _commitment_from(args)
        uid = 0
        hotkey = "(uncommitted)"
        note("checking the values you passed, not the chain")
    else:
        wallet = open_wallet(config)
        hotkey = wallet.hotkey.ss58_address
        subtensor = chain.connect(config.chain.network)
        view = chain.sync_metagraph_view(subtensor, netuid=config.chain.netuid)
        resolved = view.uid_for(hotkey)
        if resolved is None:
            raise ConfigError(
                f"hotkey {hotkey} is not registered on netuid={config.chain.netuid}; "
                "register before committing a model"
            )
        uid = resolved
        found = read_commitment(subtensor, netuid=config.chain.netuid, hotkey=hotkey, uid=uid)
        if found is None:
            raise ConfigError(
                f"hotkey {hotkey} (uid={uid}) has no model commitment on "
                f"netuid={config.chain.netuid}; run `prometheon model commit` first"
            )
        commitment = found

    with (
        HuggingFaceClient() as huggingface,
    ):
        registry = ModelRegistry(
            huggingface=huggingface,
            max_weight_bytes=config.evaluation.max_weight_bytes,
        )
        result = registry.validate([MinerEntry(uid=uid, hotkey=hotkey, commitment=commitment)])[0]

    out(
        table(
            [
                ("hotkey", result.hotkey),
                ("hf_repo", result.hf_repo),
                ("hf_revision", result.hf_revision),
                ("model_type", result.model_type or "-"),
                (
                    "weights",
                    f"{result.weight_bytes / 1024**3:.1f} GiB" if result.weight_bytes else "-",
                ),
                ("eligible", "yes" if result.valid else "no"),
            ]
        )
    )

    if result.valid:
        return EXIT_OK

    note("")
    note(f"not eligible [{result.reason.value if result.reason else 'unknown'}]: {result.detail}")
    return EXIT_FAILED
