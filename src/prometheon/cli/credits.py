"""Credit commands: read a balance, and buy credits with a stake transfer.

A model is evaluated only when its submission fee is covered by a credit. A
miner buys credits by moving alpha stake on the subnet to the treasury coldkey
(``transfer_stake``): one credit per ``submission_fee_alpha``, remainders carry
forward, and the credits belong to the sending coldkey, shared by every hotkey
under it. The hub reads those transfers from the chain and publishes the ledger
with the block and extrinsic of every entry, so anyone can recompute it.

``credits buy`` signs with the miner's own coldkey. Nothing here ever needs the
treasury's key. It prints the exact transfer first, refuses an unset or
malformed treasury address, and with ``--dry-run`` stops there.

**There are no refunds.** The command says so before anything is sent.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Final

from prometheon.chain import subtensor as chain
from prometheon.cli._common import EXIT_OK, load, note, open_wallet, out, table
from prometheon.config import Config
from prometheon.errors import ConfigError
from prometheon.hub.client import HubClient
from prometheon.hub.models import is_ss58

#: Rao per alpha (and per TAO): amounts travel as integers.
RAO_PER_ALPHA: Final[int] = 1_000_000_000

#: Where the treasury address may come from when the config leaves it empty.
TREASURY_ENV: Final[str] = "PROMETHEON_TREASURY_COLDKEY"


def _alpha(rao: int) -> str:
    whole, fraction = divmod(rao, RAO_PER_ALPHA)
    return f"{whole}.{fraction:09d}".rstrip("0").rstrip(".") + " alpha"


def _hub(config: Config) -> HubClient:
    return HubClient(
        base_url=config.hub.base_url,
        netuid=config.chain.netuid,
        epoch_keys=config.hub.epoch_keys,
        timeout_seconds=config.hub.request_timeout_seconds,
        max_retries=config.hub.max_retries,
    )


def treasury_coldkey(config: Config) -> str:
    """The treasury address, from config or the environment, checked."""
    configured = config.credits.treasury_coldkey.strip()
    from_env = os.environ.get(TREASURY_ENV, "").strip()
    if configured and from_env and configured != from_env:
        raise ConfigError(
            f"[credits] treasury_coldkey and {TREASURY_ENV} name different addresses; "
            "refusing to guess which one the fee should go to"
        )
    address = configured or from_env
    if not address:
        raise ConfigError(
            "no treasury coldkey is set. It is published with the release that launches the "
            f"subnet; set [credits] treasury_coldkey or {TREASURY_ENV}. Nothing was sent"
        )
    if not is_ss58(address):
        raise ConfigError(f"the treasury coldkey {address!r} is not an SS58 address")
    try:
        from bittensor_wallet import Keypair

        Keypair(ss58_address=address)
    except Exception as exc:
        raise ConfigError(
            f"the treasury coldkey {address!r} is not a valid address: {exc}"
        ) from exc
    return address


def cmd_balance(args: argparse.Namespace) -> int:
    """Show a coldkey's credits and recent ledger, read from the hub's public route."""
    config = load(args.config)
    address = args.address
    if not address:
        address = _wallet_coldkey(config)
    with _hub(config) as hub:
        balance = hub.get_credits(address)

    fee = balance.submission_fee_alpha_rao or config.credits.submission_fee_alpha * RAO_PER_ALPHA
    out(
        table(
            [
                ("coldkey", balance.coldkey),
                ("credits", str(balance.balance)),
                ("carried forward", _alpha(balance.remainder_alpha_rao)),
                ("fee per credit", _alpha(fee)),
            ]
        )
    )
    for entry in balance.entries[: args.limit]:
        amount = f" {_alpha(entry.alpha_rao)}" if entry.alpha_rao is not None else ""
        credits = f" {entry.credits:+d} credit(s)" if entry.credits is not None else ""
        where = f" {entry.extrinsic_hash}" if entry.extrinsic_hash else ""
        model = f" for {entry.hotkey}" if entry.hotkey else ""
        out(f"  block {entry.block}: {entry.kind}{amount}{credits}{model}{where}")
    return EXIT_OK


@chain.closes_connections
def cmd_buy(args: argparse.Namespace) -> int:
    """Buy credits: move alpha stake on the subnet to the treasury coldkey."""
    config = load(args.config)
    if args.credits < 1:
        raise ConfigError("--credits must be at least 1")
    amount_rao = args.credits * config.credits.submission_fee_alpha * RAO_PER_ALPHA
    treasury = treasury_coldkey(config)
    wallet = open_wallet(config)
    coldkey = wallet.coldkeypub.ss58_address
    hotkey = args.hotkey or wallet.hotkey.ss58_address
    if not is_ss58(hotkey):
        raise ConfigError(f"--hotkey {hotkey!r} is not an SS58 address")

    out(
        table(
            [
                ("from coldkey", coldkey),
                ("to treasury coldkey", treasury),
                ("stake on hotkey", hotkey),
                ("netuid", str(config.chain.netuid)),
                ("amount", _alpha(amount_rao)),
                ("buys", f"{args.credits} credit(s)"),
            ]
        )
    )
    note("")
    note(
        "This is a transfer_stake on chain: your alpha stake on that hotkey moves to the "
        "treasury coldkey. Credits belong to your coldkey and are spent when a new model "
        "first appears in a frozen epoch. There are no refunds."
    )

    subtensor = chain.connect(config.chain.network)
    staked = chain.read_stake_rao(
        subtensor, coldkey=coldkey, hotkey=hotkey, netuid=config.chain.netuid
    )
    note(f"your stake on that hotkey: {_alpha(staked)}")
    if staked < amount_rao:
        raise ConfigError(
            f"that hotkey holds {_alpha(staked)} of your stake on netuid "
            f"{config.chain.netuid}, less than {_alpha(amount_rao)}. Stake first, or pass "
            "--hotkey for the hotkey your stake is on. Nothing was sent"
        )

    if args.dry_run:
        note("dry run: nothing was sent")
        return EXIT_OK
    if not args.yes and not _confirmed(amount_rao, treasury):
        note("not confirmed: nothing was sent")
        return EXIT_OK

    receipt = chain.submit_transfer_stake(
        subtensor,
        wallet=wallet,
        destination_coldkey=treasury,
        hotkey=hotkey,
        netuid=config.chain.netuid,
        amount_rao=amount_rao,
    )
    note(f"sent: {receipt or 'no receipt returned'}")
    note(
        "Credits appear once the hub has indexed that block. Check with "
        "`prometheon credits balance`"
    )
    return EXIT_OK


def _confirmed(amount_rao: int, treasury: str) -> bool:
    try:
        answer = input(f"Type 'yes' to send {_alpha(amount_rao)} to {treasury}: ")
    except EOFError:
        return False
    return answer.strip().lower() == "yes"


def _wallet_coldkey(config: Config) -> str:
    wallet: Any = open_wallet(config)
    return str(wallet.coldkeypub.ss58_address)


__all__ = ["RAO_PER_ALPHA", "TREASURY_ENV", "cmd_balance", "cmd_buy", "treasury_coldkey"]
