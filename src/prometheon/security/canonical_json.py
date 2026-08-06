"""Canonical JSON encoding, and the signing domains.

Every signed object in the subnet is encoded the same way before it is hashed
or signed::

    ASCII(domain) + b"\\n" + JCS(payload)

The domain prefix is what stops a signature made for one purpose being replayed
as another. Domains are checked against :data:`ALL_DOMAINS` before encoding, so
a typo fails loudly at the call site instead of silently producing bytes nobody
else will reconstruct.

**Floats are rejected.** JSON number formatting is not canonical across
implementations, and two verifiers that format ``0.1`` differently produce
different bytes and therefore different verdicts. Every signed payload in this
subnet uses integers and strings only.
"""

from __future__ import annotations

import json
from typing import Any, Final

from prometheon.errors import CanonicalEncodingError

#: Requests from a miner or validator CLI to the subnet DB layer.
DOMAIN_DB_REQUEST: Final[str] = "PROMETHEON_V2_DB_REQUEST"

#: A validator's signed evaluation result for one cycle.
DOMAIN_EVALUATION: Final[str] = "PROMETHEON_V2_EVALUATION"

#: A miner's on-chain model commitment.
DOMAIN_MODEL_COMMITMENT: Final[str] = "PROMETHEON_V2_MODEL_COMMITMENT"

ALL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        DOMAIN_DB_REQUEST,
        DOMAIN_EVALUATION,
        DOMAIN_MODEL_COMMITMENT,
    }
)

_SEPARATOR: Final[bytes] = b"\n"


def to_canonical_bytes(payload: Any) -> bytes:
    """Encode ``payload`` as canonical JSON: sorted keys, no whitespace, UTF-8."""
    _reject_floats(payload)
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"payload is not canonicalisable: {exc}") from exc
    return text.encode("utf-8")


def domain_prefixed_bytes(domain: str, payload: Any) -> bytes:
    """The exact bytes signed or hashed for ``domain``.

    Any conforming implementation reconstructs these from the same inputs,
    which is what lets a third party verify without our code.
    """
    if domain not in ALL_DOMAINS:
        raise CanonicalEncodingError(f"unknown signing domain: {domain!r}")
    return domain.encode("ascii") + _SEPARATOR + to_canonical_bytes(payload)


def _reject_floats(value: Any, *, path: str = "$") -> None:
    """Refuse floats anywhere in a payload that is about to be signed."""
    if isinstance(value, bool):
        return
    if isinstance(value, float):
        raise CanonicalEncodingError(
            f"float at {path}: signed payloads use integers and strings only, "
            "because JSON number formatting is not canonical across implementations"
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_floats(item, path=f"{path}[{index}]")


__all__ = [
    "ALL_DOMAINS",
    "DOMAIN_DB_REQUEST",
    "DOMAIN_EVALUATION",
    "DOMAIN_MODEL_COMMITMENT",
    "domain_prefixed_bytes",
    "to_canonical_bytes",
]
