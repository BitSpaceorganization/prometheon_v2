"""Miner commands: render the wrapper, commit it, and check it before a cycle does.

The three exist as separate steps because they fail for different reasons and at
different costs. Rendering is local and free. Committing writes to the chain and
spends a slice of a per-epoch byte quota. Verification calls Hugging Face and
Chutes, and it is the only one that can tell a miner what a *validator* will
conclude, which is the answer that decides whether they earn anything.

``verify`` runs the validator's own
:class:`~prometheon.registry.validation.ModelRegistry` rather than a
reimplementation of it. A separate "miner-side check" would be a second
definition of eligibility that could agree with the first for months and then
diverge on the day it mattered.
"""

from __future__ import annotations

import argparse
import os

from prometheon.canonical.hashes import (
    ACCEPTED_WRAPPER_HASHES,
    IMAGE_NAME,
    IMAGE_TAG,
    IMAGE_USERNAME,
    chute_id_for,
    image_id_for,
)
from prometheon.canonical.integrity import (
    canonical_wrapper_hash,
    render_wrapper,
    require_valid_revision,
    wrapper_hash,
)
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
from prometheon.registry.chutes import ChutesClient
from prometheon.registry.huggingface import HuggingFaceClient
from prometheon.registry.loadability import require_loadable
from prometheon.registry.validation import MinerEntry, ModelRegistry


def _commitment_from(args: argparse.Namespace) -> ModelCommitment:
    """Build the commitment from flags, falling back to ``[miner]`` in config.

    Flags win over config so a miner can deploy a new revision without editing
    a file, but the config exists so the common case is one short command.
    """
    config = load(args.config)
    hf_repo = args.hf_repo or config.miner.hf_repo
    chute_id = args.chute_id or config.miner.chute_id
    hf_revision = args.hf_revision or config.miner.hf_revision

    missing = [
        name
        for name, value in (
            ("--hf-repo", hf_repo),
            ("--hf-revision", hf_revision),
            ("--chute-id", chute_id),
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
    return ModelCommitment(hf_repo=hf_repo, hf_revision=hf_revision, chute_id=chute_id)


def cmd_render(args: argparse.Namespace) -> int:
    """Print the exact wrapper to deploy, and verify it hashes to canonical.

    A wrapper that does not hash to an accepted value is rejected by every
    validator. The cheapest place to discover that is here, before it is
    deployed, rather than after a day of serving.
    """
    config = load(args.config)
    chutes_user = args.chutes_user or config.miner.chutes_user
    hf_repo = args.hf_repo or config.miner.hf_repo
    hf_revision = args.hf_revision or config.miner.hf_revision
    if not (chutes_user and hf_repo and hf_revision):
        raise ConfigError(
            "rendering needs --chutes-user, --hf-repo and --hf-revision "
            "(or their [miner] config equivalents)"
        )
    require_valid_revision(hf_revision)

    # The hotkey names the chute, which is what makes its id computable — and
    # therefore checkable against the commitment. A miner-chosen name left the
    # committed id unverifiable, and unknowable besides: it had to be committed
    # before any chute existed to have one.
    hotkey = open_wallet(config).hotkey.ss58_address

    # Cheapest place to catch an architecture the image cannot load. The
    # container discovers it hours later, as a startup crash with no failure
    # reason on the API -- indistinguishable from missing GPU capacity.
    with HuggingFaceClient() as huggingface:
        checked = require_loadable(huggingface, hf_repo, hf_revision)
    if checked is None:
        note(
            "transformers is not installed here, so the architecture was not checked; "
            "`uv sync --extra wrapper` installs the version the image pins"
        )

    image_id = image_id_for(IMAGE_USERNAME, IMAGE_NAME, IMAGE_TAG)
    source = render_wrapper(
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        chutes_user=chutes_user,
        hotkey=hotkey,
        image_id=image_id,
        gpu_count=args.gpu_count,
        min_vram_gb=args.min_vram_gb,
    )

    chute_id = chute_id_for(chutes_user, hotkey)
    digest = wrapper_hash(source)
    if digest not in ACCEPTED_WRAPPER_HASHES:
        raise ConfigError(
            f"the rendered wrapper hashes to {digest}, which is not an accepted "
            "value. This build's canonical assets are inconsistent — do not "
            "deploy this; reinstall the package"
        )

    if args.output:
        args.output.write_text(source, encoding="utf-8")
        note(f"wrote {args.output}")
    else:
        out(source)

    note("")
    note(
        table(
            [
                ("script sha256", digest),
                ("image", f"{IMAGE_USERNAME}/{IMAGE_NAME}:{IMAGE_TAG}"),
                ("image id", image_id),
                # Printed because it is what goes on chain, and it cannot be
                # looked up before the chute exists — it is derived from the
                # account and the hotkey, so it is knowable now.
                ("chute id (commit this)", chute_id),
                ("gpus", f"{args.gpu_count} x >= {args.min_vram_gb}GB"),
                ("accepted", "yes"),
            ]
        )
    )
    return EXIT_OK


def cmd_commit(args: argparse.Namespace) -> int:
    """Write ``(hf_repo, hf_revision, chute_id)`` on chain. This is the submission."""
    config = load(args.config)
    commitment = _commitment_from(args)
    payload = encode_commitment(commitment)
    size = commitment_size_bytes(commitment)

    out(
        table(
            [
                ("hf_repo", commitment.hf_repo),
                ("hf_revision", commitment.hf_revision),
                ("chute_id", commitment.chute_id),
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
    committed yet can pass ``--hf-repo``/``--hf-revision``/``--chute-id`` to dry
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
        # Fall back to the same env var the validator reads. A private chute
        # is invisible without a key, and the platform answers 404 rather than
        # 403 — so verifying without one reported a chute that plainly exists
        # as missing, which sent miners hunting a deployment bug they did not
        # have.
        ChutesClient(
            api_key=args.chutes_api_key or os.environ.get("CHUTES_API_KEY") or None
        ) as chutes,
    ):
        registry = ModelRegistry(huggingface=huggingface, chutes=chutes)
        result = registry.validate([MinerEntry(uid=uid, hotkey=hotkey, commitment=commitment)])[0]

    out(
        table(
            [
                ("hotkey", result.hotkey),
                ("hf_repo", result.hf_repo),
                ("hf_revision", result.hf_revision),
                ("chute_id", result.chute_id),
                ("wrapper", result.wrapper_digest or "-"),
                ("endpoint", result.moderate_url or "-"),
                ("eligible", "yes" if result.valid else "no"),
            ]
        )
    )

    if result.valid:
        return EXIT_OK

    note("")
    note(f"not eligible [{result.reason.value if result.reason else 'unknown'}]: {result.detail}")
    return EXIT_FAILED


def cmd_canonical(args: argparse.Namespace) -> int:
    """Print the hashes every deployment is verified against."""
    out(
        table(
            [
                ("wrapper sha256", canonical_wrapper_hash()),
                ("accepted wrappers", str(len(ACCEPTED_WRAPPER_HASHES))),
            ]
        )
    )
    return EXIT_OK


__all__ = ["cmd_canonical", "cmd_commit", "cmd_render", "cmd_verify"]
