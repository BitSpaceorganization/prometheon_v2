"""Behaviour of the on-chain model commitment codec."""

from __future__ import annotations

from typing import Any

import pytest

from prometheon.chain.commitment import (
    COMMITMENT_MAX_BYTES,
    ModelCommitment,
    commitment_size_bytes,
    decode_commitment,
    encode_commitment,
    publish_commitment,
    read_commitment,
    read_commitment_block,
)
from prometheon.errors import CommitmentDecodeError, CommitmentError
from prometheon.registry.validation import UNKNOWN_COMMIT_BLOCK

pytestmark = pytest.mark.unit

REVISION = "a0243b47c1f9e8d2b6045a7c3e1f8b9d0c2a4e6f"
REPO = "prometheon-labs/moderation-guard-8b"


def realistic() -> ModelCommitment:
    return ModelCommitment(hf_repo=REPO, hf_revision=REVISION)


class TestSizeBudget:
    def test_realistic_commitment_fits_the_chain_budget(self) -> None:
        size = commitment_size_bytes(realistic())
        assert size <= COMMITMENT_MAX_BYTES
        # Headroom matters: a longer repo name must not push a miner over.
        # Format 2 costs 30 fixed bytes where format 1 cost 53, because the
        # deployment id and its mode byte are gone.
        assert size == 65

    def test_the_encoding_still_earns_its_keep_against_naive_json(self) -> None:
        """Two fields do fit as JSON, which three did not. That is not the case
        for keeping the encoding.

        The budget is charged twice: the chain refuses anything over 128 bytes,
        and the commitments pallet also bills every write against a per-hotkey
        quota that resets each epoch. A payload 40% smaller buys more
        corrections on the day a miner needs one, and leaves room for a repo id
        that JSON would push over.
        """
        import json

        naive = json.dumps(
            {"hf_repo": REPO, "hf_revision": REVISION},
            sort_keys=True,
            separators=(",", ":"),
        )
        encoded = commitment_size_bytes(realistic())

        assert encoded < len(naive.encode())
        assert encoded < len(naive.encode()) * 0.7

    def test_a_repo_filling_the_whole_budget_still_fits(self) -> None:
        repo = "a" * 48 + "/" + "b" * 49  # 98 characters, the remaining room
        assert len(repo) == 98
        commitment = ModelCommitment(hf_repo=repo, hf_revision=REVISION)
        assert commitment_size_bytes(commitment) == COMMITMENT_MAX_BYTES

    def test_oversize_repo_is_rejected_with_the_overshoot_named(self) -> None:
        repo = "a" * 60 + "/" + "b" * 60
        commitment = ModelCommitment(hf_repo=repo, hf_revision=REVISION)
        with pytest.raises(CommitmentError, match="over the chain's 128-byte limit"):
            encode_commitment(commitment)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "repo",
        ["org/name", "single", "a" * 40 + "/" + "b" * 40, "org/name.with.dots", "org/name-1_2"],
    )
    def test_repo_survives_a_round_trip(self, repo: str) -> None:
        original = ModelCommitment(hf_repo=repo, hf_revision=REVISION)
        assert decode_commitment(encode_commitment(original)).hf_repo == repo

    @pytest.mark.parametrize(
        "revision",
        ["0" * 40, "f" * 40, REVISION, "0123456789abcdef" * 2 + "01234567"],
    )
    def test_revision_survives_a_round_trip(self, revision: str) -> None:
        original = ModelCommitment(hf_repo=REPO, hf_revision=revision)
        assert decode_commitment(encode_commitment(original)).hf_revision == revision

    def test_encoding_is_canonical(self) -> None:
        """One commitment, one payload: re-encoding a decode changes nothing."""
        payload = encode_commitment(realistic())
        assert encode_commitment(decode_commitment(payload)) == payload

    def test_bytes_and_hex_forms_decode_identically(self) -> None:
        payload = encode_commitment(realistic())
        expected = realistic()
        assert decode_commitment(payload.encode()) == expected
        assert decode_commitment("0x" + payload.encode().hex()) == expected


class TestConstructionValidation:
    @pytest.mark.parametrize(
        "revision",
        [
            "main",
            "v1.0",
            "A0243B47C1F9E8D2B6045A7C3E1F8B9D0C2A4E6F",  # uppercase hex
            "a0243b47",
            "a0243b47c1f9e8d2b6045a7c3e1f8b9d0c2a4e6",  # 39 chars
            "",
        ],
    )
    def test_a_moving_reference_is_not_a_commitment(self, revision: str) -> None:
        with pytest.raises(CommitmentError, match="40-character lowercase hex SHA"):
            ModelCommitment(hf_repo=REPO, hf_revision=revision)

    @pytest.mark.parametrize(
        "repo", ["", "-leading-dash", "a/b/c", "org/na me", "org/name:tag", "/name", "org/"]
    )
    def test_malformed_repo_is_rejected(self, repo: str) -> None:
        with pytest.raises(CommitmentError, match="hf_repo"):
            ModelCommitment(hf_repo=repo, hf_revision=REVISION)


class TestTheLegacyFormatIsRefused:
    """Format 1 named a hosted deployment. Reading its repo anyway would score a
    model the miner never resubmitted under the rules that now apply."""

    def test_a_format_1_payload_is_refused_by_name(self) -> None:
        payload = "P21uQMBpgk9CUake768oHr5MVE79PhgfRYn8ey3VWmdZnF-IvhO1QQowner/model"
        with pytest.raises(CommitmentDecodeError, match="format 1"):
            decode_commitment(payload)

    def test_the_refusal_says_what_to_do_about_it(self) -> None:
        payload = "P21uQMBpgk9CUake768oHr5MVE79PhgfRYn8ey3VWmdZnF-IvhO1QQowner/model"
        with pytest.raises(CommitmentDecodeError, match="re-commit"):
            decode_commitment(payload)


class TestDecodeRejection:
    def test_empty_payload(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="empty"):
            decode_commitment("")

    def test_foreign_payload(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="not a Prometheon V2"):
            decode_commitment('{"hf_repo": "org/name"}')

    def test_future_format_version_names_the_remedy(self) -> None:
        payload = encode_commitment(realistic())
        future = payload[:2] + "9" + payload[3:]
        with pytest.raises(CommitmentDecodeError, match="upgrade the validator"):
            decode_commitment(future)

    def test_truncated_header(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="truncated"):
            decode_commitment("P21u" + "A" * 10)

    def test_missing_repo(self) -> None:
        payload = encode_commitment(realistic())
        with pytest.raises(CommitmentDecodeError, match="no hf_repo"):
            decode_commitment(payload[:30])

    def test_repo_that_would_not_be_fetchable(self) -> None:
        payload = encode_commitment(realistic())
        with pytest.raises(CommitmentDecodeError, match="invalid values"):
            decode_commitment(payload[:53] + "not a repo")

    def test_non_ascii_payload(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="not ASCII"):
            decode_commitment(b"P21u" + b"\xff" * 30)

    def test_over_budget_payload(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="over the 128-byte limit"):
            decode_commitment("P21u" + "A" * 200)

    def test_revision_field_outside_the_base64url_alphabet(self) -> None:
        payload = encode_commitment(realistic())
        with pytest.raises(CommitmentDecodeError, match="hf_revision is not valid base64url"):
            decode_commitment(payload[:4] + "*" + payload[5:])

    def test_non_canonical_base64_padding_bits_are_rejected(self) -> None:
        """The last character of a 20-byte field carries bits that must be zero."""
        payload = encode_commitment(realistic())
        last = payload[29]
        # The 27th and final character of the revision field holds 4
        # significant bits and 2 that must be zero.
        # 'B' and 'C' both set one of those two, so each decodes to the same 20
        # bytes while being a second spelling of them.
        alternative = "B" if last != "B" else "C"
        mutated = payload[:29] + alternative + payload[30:]
        with pytest.raises(CommitmentDecodeError, match="canonically base64url"):
            decode_commitment(mutated)

    def test_hex_form_that_is_not_ascii_underneath(self) -> None:
        with pytest.raises(CommitmentDecodeError, match="hex-encoded"):
            decode_commitment("0xfffefdfc")


class AccessorSubtensor:
    """The pinned SDK shape: named accessors, commitments keyed by uid."""

    def __init__(self, *, commitment: Any = "", set_result: Any = True) -> None:
        self.commitment = commitment
        self.set_result = set_result
        self.written: list[str] = []
        self.read_uids: list[int] = []

    def set_commitment(self, *, wallet: Any, netuid: int, data: str) -> Any:
        self.written.append(data)
        if isinstance(self.set_result, Exception):
            raise self.set_result
        return self.set_result

    def get_commitment(self, *, netuid: int, uid: int) -> Any:
        self.read_uids.append(uid)
        if isinstance(self.commitment, Exception):
            raise self.commitment
        return self.commitment


class CommitmentRecord:
    """The newer SDK's read result: the payload wrapped in chain metadata."""

    def __init__(self, data: Any) -> None:
        self.block = 100
        self.data = data
        self.fields: list[Any] = []


class ReadRegistrySubtensor:
    """The newer SDK shape: generic reads, commitments keyed by hotkey."""

    def __init__(self, *, commitment: Any = None) -> None:
        self.commitment = commitment
        self.reads: list[tuple[str, dict[str, Any]]] = []

    def read(self, name: str, **params: Any) -> Any:
        self.reads.append((name, params))
        if isinstance(self.commitment, Exception):
            raise self.commitment
        return self.commitment

    def execute(self, intent: Any, wallet: Any) -> Any:
        raise AssertionError("not exercised here")


class TestPublish:
    def test_publish_writes_the_encoded_payload_and_returns_it(self) -> None:
        subtensor = AccessorSubtensor()
        payload = publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())
        assert subtensor.written == [payload]
        assert decode_commitment(payload) == realistic()

    def test_publish_raises_when_the_chain_reports_rejection(self) -> None:
        subtensor = AccessorSubtensor(set_result=False)
        with pytest.raises(CommitmentError, match="byte quota"):
            publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())

    def test_a_result_object_reporting_failure_is_not_read_as_success(self) -> None:
        class Result:
            success = False
            message = "SpaceLimitExceeded"

        subtensor = AccessorSubtensor(set_result=Result())
        with pytest.raises(CommitmentError, match="byte quota"):
            publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())

    @pytest.mark.parametrize("ok", [True, None, "0xabc"])
    def test_publish_accepts_every_success_shape(self, ok: Any) -> None:
        subtensor = AccessorSubtensor(set_result=ok)
        publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())
        assert len(subtensor.written) == 1

    def test_publish_wraps_sdk_exceptions(self) -> None:
        subtensor = AccessorSubtensor(set_result=TimeoutError("no response"))
        with pytest.raises(CommitmentError, match="could not publish"):
            publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())

    def test_an_sdk_that_cannot_publish_says_so_and_names_the_pin(self) -> None:
        with pytest.raises(CommitmentError, match=r"no Subtensor\.set_commitment") as excinfo:
            publish_commitment(object(), wallet=object(), netuid=42, commitment=realistic())
        assert "bittensor>=10.5,<11" in str(excinfo.value)

    def test_a_failure_tuple_is_not_read_as_success(self) -> None:
        """The shape that used to slip through.

        A ``(False, "reason")`` tuple has no ``.success``, so a check written
        as ``getattr(result, "success", True)`` defaults it to True and reports
        a rejected commitment as a published one. The miner would then wait a
        day to be scored on a model the chain never recorded.
        """
        subtensor = AccessorSubtensor(set_result=(False, "SpaceLimitExceeded"))
        with pytest.raises(CommitmentError, match="SpaceLimitExceeded"):
            publish_commitment(subtensor, wallet=object(), netuid=42, commitment=realistic())


class TestReadOnAnAccessorSdk:
    @pytest.mark.parametrize("empty", ["", "   ", b"", None])
    def test_never_committed_reads_as_none(self, empty: Any) -> None:
        subtensor = AccessorSubtensor(commitment=empty)
        assert read_commitment(subtensor, netuid=42, hotkey="5Alice", uid=7) is None

    def test_read_decodes_a_present_commitment(self) -> None:
        subtensor = AccessorSubtensor(commitment=encode_commitment(realistic()))
        assert read_commitment(subtensor, netuid=42, hotkey="5Alice", uid=7) == realistic()
        assert subtensor.read_uids == [7]

    def test_present_but_unreadable_raises_rather_than_reading_as_absent(self) -> None:
        subtensor = AccessorSubtensor(commitment="some other subnet's payload")
        with pytest.raises(CommitmentDecodeError):
            read_commitment(subtensor, netuid=42, hotkey="5Alice", uid=7)

    def test_read_wraps_sdk_exceptions(self) -> None:
        subtensor = AccessorSubtensor(commitment=ConnectionError("socket closed"))
        with pytest.raises(CommitmentError, match="could not read the commitment"):
            read_commitment(subtensor, netuid=42, hotkey="5Alice", uid=7)

    def test_the_uid_is_required_rather_than_defaulted(self) -> None:
        """A UID is a recycled slot, so it may never be guessed.

        The SDK accessor is keyed by UID. Making the argument mandatory pushes
        the hotkey → UID resolution onto the caller, which is the only place
        that knows *which snapshot* the resolution belongs to. Defaulting it
        would silently read whatever neuron currently occupies a slot, and a
        deregistration mid-cycle would attribute one miner's model to another.
        """
        subtensor = AccessorSubtensor(commitment=encode_commitment(realistic()))
        with pytest.raises(TypeError, match="uid"):
            read_commitment(subtensor, netuid=42, hotkey="5Alice")  # type: ignore[call-arg]

    def test_the_uid_the_caller_resolved_is_the_one_that_is_read(self) -> None:
        subtensor = AccessorSubtensor(commitment=encode_commitment(realistic()))
        read_commitment(subtensor, netuid=42, hotkey="5Alice", uid=7)
        assert subtensor.read_uids == [7]


class _MetadataSubtensor:
    """A node that answers ``get_commitment_metadata`` with a chosen shape."""

    def __init__(self, metadata: Any) -> None:
        self.metadata = metadata
        self.asked: list[tuple[int, str]] = []

    def get_commitment_metadata(self, *, netuid: int, hotkey_ss58: str) -> Any:
        self.asked.append((netuid, hotkey_ss58))
        if isinstance(self.metadata, Exception):
            raise self.metadata
        return self.metadata


class TestReadingTheCommitmentBlock:
    """The value that settles a duplicate-model contest.

    Every one of these is about the same failure: an unreadable block is a
    *legitimate* value (``UNKNOWN_COMMIT_BLOCK``), so getting it wrong is
    silent. The whole subnet sorted by uid for a while because of that.
    """

    def test_the_mapping_shape_the_live_sdk_actually_returns_is_read(self) -> None:
        """bittensor 10.5.0 answers with a plain ``dict``.

        The accessor's declared return type is ``str | CommitmentOfResponse``
        and the contract test asserts ``CommitmentOfResponse`` carries a
        ``block``, so the type looked covered. Against a real node it hands back
        ``{'deposit': ..., 'block': ..., 'info': ...}``. Attribute access on
        that misses, and every hotkey on the subnet read as unknown.
        """
        subtensor = _MetadataSubtensor({"deposit": 0, "block": 8926991, "info": {"fields": []}})

        assert read_commitment_block(subtensor, netuid=108, hotkey="5Alice") == 8926991

    def test_the_object_shape_still_works(self) -> None:
        class Response:
            block = 4242

        assert read_commitment_block(_MetadataSubtensor(Response()), netuid=42, hotkey="5A") == 4242

    def test_a_mapping_without_a_block_is_unknown_rather_than_zero(self) -> None:
        """Zero is a real block number and would win every contest."""
        subtensor = _MetadataSubtensor({"deposit": 0, "info": {"fields": []}})

        assert read_commitment_block(subtensor, netuid=42, hotkey="5A") == UNKNOWN_COMMIT_BLOCK

    def test_the_plain_string_shape_carries_no_block(self) -> None:
        assert (
            read_commitment_block(_MetadataSubtensor("0xdeadbeef"), netuid=42, hotkey="5A")
            == UNKNOWN_COMMIT_BLOCK
        )

    def test_an_unreadable_block_loses_rather_than_wins(self) -> None:
        """``UNKNOWN_COMMIT_BLOCK`` must sort after every real block."""
        subtensor = _MetadataSubtensor({"block": None})

        unknown = read_commitment_block(subtensor, netuid=42, hotkey="5A")
        assert unknown == UNKNOWN_COMMIT_BLOCK
        assert unknown > 8_926_991

    def test_it_is_keyed_by_hotkey_not_uid(self) -> None:
        """A uid is a recycled slot; a hotkey is identity."""
        subtensor = _MetadataSubtensor({"block": 7})
        read_commitment_block(subtensor, netuid=108, hotkey="5Bob")
        assert subtensor.asked == [(108, "5Bob")]
