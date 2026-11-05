"""Ed25519 verification is plain RFC 8032, so the hub's Node signatures verify.

The vectors are RFC 8032 section 7.1, tests 1 and 2.
"""

from __future__ import annotations

import pytest

from prometheon.errors import SignatureFormatError, SignatureVerificationError
from prometheon.security.signatures import verify_ed25519

pytestmark = pytest.mark.unit

RFC_1 = (
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
    b"",
    "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bac"
    "c61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
)
RFC_2 = (
    "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
    bytes.fromhex("72"),
    "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e"
    "458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
)


@pytest.mark.parametrize("vector", [RFC_1, RFC_2], ids=["rfc8032-test1", "rfc8032-test2"])
def test_the_rfc_vectors_verify(vector: tuple[str, bytes, str]) -> None:
    public_key, message, signature = vector
    verify_ed25519(public_key_hex=public_key, signature_hex=signature, message=message)
    verify_ed25519(public_key_hex=public_key, signature_hex="0x" + signature, message=message)


def test_a_changed_message_does_not_verify() -> None:
    public_key, _message, signature = RFC_2
    with pytest.raises(SignatureVerificationError):
        verify_ed25519(public_key_hex=public_key, signature_hex=signature, message=b"s")


def test_another_key_does_not_verify() -> None:
    _key, message, signature = RFC_2
    with pytest.raises(SignatureVerificationError):
        verify_ed25519(public_key_hex=RFC_1[0], signature_hex=signature, message=message)


@pytest.mark.parametrize(
    ("public_key", "signature"),
    [
        ("D75A" + "0" * 60, RFC_1[2]),
        ("00" * 31, RFC_1[2]),
        (RFC_1[0], RFC_1[2][:-2]),
        (RFC_1[0], RFC_1[2].upper()),
    ],
)
def test_malformed_keys_and_signatures_are_refused(public_key: str, signature: str) -> None:
    with pytest.raises(SignatureFormatError):
        verify_ed25519(public_key_hex=public_key, signature_hex=signature, message=b"")
