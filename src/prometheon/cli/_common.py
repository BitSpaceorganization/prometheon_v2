"""Shared plumbing for every command: config, wallet, clients, output.

Kept apart from the commands so each command file is about what it does rather
than about how it got a keypair. Two rules hold everywhere in this package:

**Nothing here prints a secret.** Wallets are opened through ``bittensor_wallet``
and only ever produce a public ss58 address for display. API keys are read from
the environment by name and are never echoed, not even truncated. A prefix is
enough to confirm a leak.

**Every failure is a typed error with a code.** ``main`` renders those into one
line and an exit status. If a traceback reaches an operator, this package missed
a case; the operator did nothing exotic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Final

from prometheon.config import Config, load_config
from prometheon.errors import ConfigError, PrometheonError

#: Exit codes. Distinct so a supervisor can tell "this will never work" from
#: "try again later" without parsing text.
EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_USAGE: Final[int] = 2


def load(path: str | Path) -> Config:
    """Load a config file, or raise :class:`ConfigError` naming the file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise ConfigError(
            f"no config file at {resolved}. Copy configs/testnet.example.toml and "
            "edit the [wallet] section"
        )
    return load_config(resolved)


def open_wallet(config: Config) -> Any:
    """Open the configured wallet and return its hotkey keypair.

    Raises rather than returning ``None``: every command that calls this needs
    to sign something, and a missing wallet must stop the command before it
    does half its work.
    """
    try:
        from bittensor_wallet import Wallet
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ConfigError(
            f"bittensor-wallet is not installed, so no hotkey can be loaded: {exc}"
        ) from exc

    try:
        wallet = Wallet(name=config.wallet.name, hotkey=config.wallet.hotkey)
        keypair = wallet.hotkey
    except Exception as exc:
        raise ConfigError(
            f"could not open wallet name={config.wallet.name!r} "
            f"hotkey={config.wallet.hotkey!r}: {exc}. Check `btcli wallet list`"
        ) from exc

    if keypair is None:
        raise ConfigError(f"wallet {config.wallet.name!r} has no hotkey {config.wallet.hotkey!r}")
    return wallet


def require_env(name: str, *, purpose: str) -> str:
    """Read a required environment variable, or explain what it is for.

    The value is never echoed back, on any path.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"environment variable {name} is not set. It is required to {purpose}")
    return value


def out(message: str = "") -> None:
    """Write to stdout. Results go here so they can be piped."""
    print(message, file=sys.stdout)


def note(message: str) -> None:
    """Write progress to stderr, so it never contaminates piped output."""
    print(message, file=sys.stderr)


def fail(exc: PrometheonError) -> int:
    """Render a typed failure as one greppable line and return an exit code."""
    note(f"error [{exc.code}] {exc}")
    return EXIT_FAILED


def table(rows: list[tuple[str, str]]) -> str:
    """Two aligned columns. Used for every summary this CLI prints."""
    if not rows:
        return ""
    width = max(len(key) for key, _ in rows)
    return "\n".join(f"{key.ljust(width)}  {value}" for key, value in rows)


__all__ = [
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "fail",
    "load",
    "note",
    "open_wallet",
    "out",
    "require_env",
    "table",
]
