"""Signatures: SR25519 for hotkeys, Ed25519 for the hub's epoch.

Identity in this subnet is a Bittensor hotkey and nothing else. There are no
API keys to issue, rotate, or leak: a caller proves who it is by signing, and
the verifier checks the signature against the hotkey the payload names. A
deregistered hotkey loses access automatically because the permit check fails,
not because someone remembered to revoke a credential.
"""

from __future__ import annotations

import re
from typing import Any, Final

from bittensor_wallet import Keypair

from prometheon.errors import (
    SignatureFormatError,
    SignatureVerificationError,
)
from prometheon.security.canonical_json import domain_prefixed_bytes

#: ``0x`` + 128 lowercase hex characters.
_SIGNATURE_RE: Final[re.Pattern[str]] = re.compile(r"^0x[0-9a-f]{128}$")

#: Loose SS58 shape check. Full validation happens when the keypair is built.
_SS58_RE: Final[re.Pattern[str]] = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,48}$")

#: A raw Ed25519 public key: 32 bytes as 64 lowercase hex characters.
_ED25519_KEY_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

#: ``bittensor_wallet``'s name for Ed25519. Plain RFC 8032 Ed25519, checked
#: against the RFC's test vectors in the tests, so a signature made by Node's
#: ``crypto.sign`` verifies here unchanged.
_CRYPTO_TYPE_ED25519: Final[int] = 0


def sign(keypair: Keypair, domain: str, payload: Any) -> str:
    """Sign the domain-prefixed canonical bytes of ``payload``.

    Returns ``0x``-prefixed lowercase hex. The keypair must hold a private key.
    """
    message = domain_prefixed_bytes(domain, payload)
    return "0x" + keypair.sign(message).hex()


def verify(*, ss58_address: str, signature_hex: str, domain: str, payload: Any) -> None:
    """Verify a signature, raising on any failure.

    Raises rather than returning a boolean: a caller that forgets to check a
    returned ``False`` silently accepts forged input, and that mistake has
    shipped in more than one codebase. There is no way to ignore an exception
    by accident.
    """
    if not _SIGNATURE_RE.match(signature_hex or ""):
        raise SignatureFormatError(
            "signature must be '0x' followed by 128 lowercase hex characters"
        )
    if not _SS58_RE.match(ss58_address or ""):
        raise SignatureFormatError(f"not an SS58 address: {ss58_address!r}")

    message = domain_prefixed_bytes(domain, payload)
    try:
        keypair = Keypair(ss58_address=ss58_address)
        ok = keypair.verify(message, bytes.fromhex(signature_hex[2:]))
    except SignatureFormatError:
        raise
    except Exception as exc:
        raise SignatureVerificationError(f"signature could not be verified: {exc}") from exc

    if not ok:
        raise SignatureVerificationError(
            f"signature does not verify against {ss58_address} under {domain}"
        )


def is_ed25519_public_key(value: str) -> bool:
    return bool(_ED25519_KEY_RE.match(value or ""))


def verify_ed25519(*, public_key_hex: str, signature_hex: str, message: bytes) -> None:
    """Verify a raw Ed25519 signature over ``message``, raising on any failure.

    The hub signs its epoch with Ed25519, the same scheme as the BitFan feed,
    over canonical JSON with no domain prefix: the key signs nothing else. The
    signature is accepted with or without a ``0x`` prefix, because that is
    formatting rather than content.
    """
    if not is_ed25519_public_key(public_key_hex):
        raise SignatureFormatError(
            "an Ed25519 public key must be 64 lowercase hex characters (32 raw bytes)"
        )
    body = signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
    if not re.fullmatch(r"[0-9a-f]{128}", body or ""):
        raise SignatureFormatError(
            "an Ed25519 signature must be 128 lowercase hex characters, optionally 0x-prefixed"
        )
    try:
        keypair = Keypair(public_key=public_key_hex, crypto_type=_CRYPTO_TYPE_ED25519)
        ok = keypair.verify(message, bytes.fromhex(body))
    except Exception as exc:
        raise SignatureVerificationError(f"signature could not be verified: {exc}") from exc
    if not ok:
        raise SignatureVerificationError("Ed25519 signature does not verify against the pinned key")


__all__ = ["is_ed25519_public_key", "sign", "verify", "verify_ed25519"]
