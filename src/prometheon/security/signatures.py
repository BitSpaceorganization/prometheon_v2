"""SR25519 signing and verification over domain-prefixed canonical bytes.

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


__all__ = ["sign", "verify"]
