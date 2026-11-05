"""Byte-level vectors shared with the hub, which is written in another language.

The epoch below was put into canonical form by this package and signed by
Node's ``crypto.sign`` with the first test key of RFC 8032, so the signature
comes from a different Ed25519 implementation than the one validators run. A
JavaScript canonicaliser (keys sorted at every level, compact
``JSON.stringify``) produced byte-identical input, including the non-ASCII
text and the U+2028 in ``note``.

If any test here fails, this client and the hub no longer agree on the wire,
and every validator would refuse every real epoch. Change these values only
together with the hub, and record why in the wire notes.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest

from prometheon.errors import EpochVerificationError
from prometheon.hub.epoch import epoch_content_hash, epoch_signing_bytes, verify_epoch
from prometheon.hub.fake import FakeHub
from prometheon.registry.huggingface import RepoFile, RepoSnapshot, weights_fingerprint

pytestmark = pytest.mark.contract

#: RFC 8032 section 7.1, test 1: the public half of the fake hub's test key.
RFC8032_TEST_1_PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"

EPOCH: dict[str, Any] = {
    "schema": "prometheon-epoch/1",
    "netuid": 1,
    "day": "2026-11-05",
    "frozen_block": 123456,
    "scoring_version": "prometheon-scoring/3.0",
    "policy": {
        "version": "2026-08-07",
        "sha256": "0f50750208fbc84499c8c7c1017a3f8c1f02eb482db12aa2f79da4b21d69157f",
    },
    "miners": [
        {
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "uid": 3,
            "coldkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "verified_active_members": 14,
            "members": 40,
            "eligible": True,
        }
    ],
    "models": [
        {
            "hotkey": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            "repo": "org/name",
            "revision": "a" * 40,
            "commit_block": 1,
            "fingerprint": "b" * 64,
            "payment": "paid",
        }
    ],
    "files": {
        "test": {"sha256": "c" * 64, "count": 1},
        "production": {"sha256": "d" * 64, "count": 1},
    },
    "key_id": "hub-2026-11",
    # A field this client does not know: ignored when read, covered when signed.
    "note": "caf\u00e9 \u2028",
}

#: sha256 of the canonical bytes of EPOCH, which is also its content hash.
SIGNED_BYTES_SHA256 = "8ca6ef49624dac6caf8447ccd415bcbcd24494cdf4337e50faa9e2800f29e253"

#: Node 24 ``crypto.sign(null, bytes, key)`` with the RFC 8032 test 1 seed.
NODE_SIGNATURE = (
    "0x3bbc8c4c538cf59aa0da9b451f6eee13b24862ce344b9b1dc7f1998242b900c6"
    "96ae0d6b13a0e166c1b2bdb5371125e3cdaaf76f3e5327336ce518a544d2f203"
)

KEYS = {"hub-2026-11": RFC8032_TEST_1_PUBLIC_KEY}


def _signed() -> dict[str, Any]:
    return {**copy.deepcopy(EPOCH), "signature": NODE_SIGNATURE}


def test_the_canonical_bytes_are_the_ones_node_signed() -> None:
    assert hashlib.sha256(epoch_signing_bytes(_signed())).hexdigest() == SIGNED_BYTES_SHA256


def test_a_signature_made_by_node_verifies_here() -> None:
    epoch = verify_epoch(_signed(), keys=KEYS)
    assert (epoch.miners[0].verified_active_members, epoch.miners[0].members) == (14, 40)


def test_the_content_hash_is_the_sha256_of_the_signed_bytes() -> None:
    assert epoch_content_hash(_signed()) == SIGNED_BYTES_SHA256


def test_the_fake_hub_signs_exactly_as_node_does() -> None:
    hub = FakeHub(netuid=1, key_id="hub-2026-11")
    assert hub.public_key == RFC8032_TEST_1_PUBLIC_KEY
    assert hub.sign(EPOCH) == NODE_SIGNATURE


def test_the_unknown_field_is_covered_by_the_signature() -> None:
    tampered = _signed()
    tampered["note"] = "cafe \u2028"
    with pytest.raises(EpochVerificationError):
        verify_epoch(tampered, keys=KEYS)


def test_the_weights_fingerprint_formula() -> None:
    """sha256 of the sorted LFS sha256s, newline-joined; names and order ignored."""
    shard_1 = hashlib.sha256(b"shard-1").hexdigest()
    shard_2 = hashlib.sha256(b"shard-2").hexdigest()
    entries = (
        RepoFile("config.json", 700, None),
        RepoFile("weights-b.safetensors", 10, shard_2),
        RepoFile("weights-a.safetensors", 10, shard_1),
    )
    snapshot = RepoSnapshot(
        repo="org/name",
        revision="a" * 40,
        files=tuple(entry.name for entry in entries),
        entries=entries,
    )
    assert (
        weights_fingerprint(snapshot)
        == "c40af48ed17b2f5cd02e1b9ffc013b5056f0152950b2df43ee3873b0dd1ad3cd"
    )
