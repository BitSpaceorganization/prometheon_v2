"""Pins the bittensor SDK shapes ``prometheon.chain`` actually depends on.

The unit tests drive the adapter through fakes, which proves the adapter's
logic but not that the fakes resemble the SDK. Agreeing mocks have hidden real
defects in this codebase before. These tests build **real** SDK objects with no
network, no chain and no wallet, then push them through the same code paths, so
a release that renames a field fails here instead of at 04:00 UTC.

They go through the adapter, not around it. An earlier version of this file
handed a hand-built object straight to ``MetagraphView.from_sdk_metagraph`` and
never called ``sync_metagraph_view``, so it pinned a type the adapter never
receives and passed while the real path returned an empty snapshot. Every test
here that can drive a public adapter function does, with a fake client whose
only job is to hand back a genuine SDK object.

Written that way, these tests immediately caught a live bug. A 10.x
``Metagraph`` carries ``neurons = []`` until it is synced with ``lite=False``,
the transposer branched on ``neurons is not None``, and a fully populated
two-neuron metagraph transposed to a **zero-neuron view** with no error raised.

Everything asserted below is offline: object construction and signature
introspection only.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from typing import Any

import pytest

from prometheon.chain import subtensor as adapter
from prometheon.chain.commitment import ModelCommitment, publish_commitment, read_commitment
from prometheon.errors import CommitmentError, SetWeightsFailedError, SubtensorError

pytestmark = pytest.mark.contract

bittensor = pytest.importorskip("bittensor", reason="the chain SDK is an optional test dependency")

REVISION = "a" * 40


def _axon(hotkey: str) -> Any:
    # The address is inert: this object is only ever read for its hotkey, and
    # nothing in these tests binds a socket or opens a connection.
    return bittensor.AxonInfo(
        version=0,
        ip="0.0.0.0",  # noqa: S104
        port=0,
        ip_type=4,
        hotkey=hotkey,
        coldkey="5Cold",
    )


def _metagraph(
    *,
    netuid: int = 42,
    hotkeys: list[str],
    uids: list[int],
    stake_tao: list[float],
    permits: list[bool],
    block: int = 1234,
) -> Any:
    """A real ``bittensor.Metagraph``, populated the way a sync populates one."""
    import numpy as np

    raw = bittensor.Metagraph(netuid=netuid, sync=False, lite=False)
    raw.axons = [_axon(hotkey) for hotkey in hotkeys]
    raw.uids = np.array(uids)
    raw.validator_permit = np.array(permits)
    raw.stake = np.array(stake_tao, dtype=np.float64)
    raw.block = np.array(block)
    return raw


class _Client:
    """The narrowest thing that looks like a ``Subtensor`` to the adapter."""

    def __init__(self, **returns: Any) -> None:
        self._returns = returns
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def metagraph(self, netuid: int, lite: bool = True) -> Any:
        self.calls.append(("metagraph", {"netuid": netuid, "lite": lite}))
        return self._returns["metagraph"]

    def get_subnet_hyperparameters(self, netuid: int) -> Any:
        return self._returns["hyperparameters"]

    def get_subnet_owner_hotkey(self, netuid: int) -> Any:
        return self._returns["owner"]

    def get_commitment(self, netuid: int, uid: int) -> Any:
        self.calls.append(("get_commitment", {"netuid": netuid, "uid": uid}))
        return self._returns["commitment"]

    def set_weights(self, **kwargs: Any) -> Any:
        self.calls.append(("set_weights", kwargs))
        return self._returns["set_weights"]

    def set_commitment(self, **kwargs: Any) -> Any:
        self.calls.append(("set_commitment", kwargs))
        return self._returns["set_commitment"]


class _Receipt:
    """The receipt surface the adapter reads, as this SDK actually shapes it.

    Just the two members ``_best_receipt`` touches, so the adapter's preference
    order is pinned against the real method name rather than an invented one.
    """

    def __init__(self, identifier: str | None = None, index: int | None = None) -> None:
        self._identifier = identifier
        self.extrinsic_idx = index

    def get_extrinsic_identifier(self) -> str | None:
        return self._identifier


def _response(**overrides: Any) -> Any:
    """A real ``ExtrinsicResponse`` with only the fields under test supplied.

    Unknown names are rejected here. A tolerant version of this helper silently
    dropped ``extrinsic_id=...``, a field this SDK does not have, and the test
    still passed by reading the status message instead. That hid the fact that
    the adapter could never produce a real receipt.
    """
    cls = bittensor.core.types.ExtrinsicResponse
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = set(overrides) - known
    assert not unknown, (
        f"ExtrinsicResponse has no field(s) {sorted(unknown)}; this SDK exposes {sorted(known)}"
    )

    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name in overrides:
            kwargs[field.name] = overrides[field.name]
        elif field.default is dataclasses.MISSING and (
            field.default_factory is dataclasses.MISSING  # type: ignore[misc]
        ):
            kwargs[field.name] = None
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# The generation gate
# ---------------------------------------------------------------------------


def test_the_installed_sdk_is_one_this_runtime_drives() -> None:
    """Every accessor the adapter calls exists on the pinned SDK.

    The check that would have caught the 11.x mis-pin at install time rather
    than at the first cycle.
    """
    adapter.assert_sdk_compatible(bittensor.Subtensor)


def test_an_11x_style_client_is_rejected_by_name() -> None:
    """A read/intent client must be refused with an explanation rather than driven."""

    class ReadIntentClient:
        def read(self, name: str, **params: Any) -> Any: ...
        def execute(self, intent: Any, wallet: Any) -> Any: ...

    with pytest.raises(SubtensorError, match=r"read.*intent|11"):
        adapter.assert_sdk_compatible(ReadIntentClient())


def test_the_installed_sdk_lets_us_name_the_mechanism() -> None:
    """A build that cannot take a mechid would fail the submission gate."""
    capabilities = adapter.detect_capabilities()
    assert capabilities.supports_mechid is True
    assert capabilities.sdk_version != "not-installed"


# ---------------------------------------------------------------------------
# Metagraph, driven through the adapter
# ---------------------------------------------------------------------------


def test_a_real_metagraph_transposes_through_the_adapter() -> None:
    """The whole path, from SDK object through the adapter to the view."""
    client = _Client(
        metagraph=_metagraph(
            hotkeys=["5Alice", "5Bob"], uids=[4, 9], stake_tao=[1.0, 7e-9], permits=[True, False]
        )
    )

    view = adapter.sync_metagraph_view(client, netuid=42)

    assert view.uids == (4, 9)
    assert view.hotkeys == ("5Alice", "5Bob")
    assert view.validator_permit == (True, False)
    assert view.block == 1234
    assert view.uid_for("5Bob") == 9


def test_the_adapter_asks_for_the_full_table_not_the_lite_one() -> None:
    """A lite metagraph omits stake and permits; permits gate validator reads."""
    client = _Client(
        metagraph=_metagraph(hotkeys=["5A"], uids=[0], stake_tao=[1.0], permits=[True])
    )
    adapter.sync_metagraph_view(client, netuid=42)
    assert client.calls[0] == ("metagraph", {"netuid": 42, "lite": False})


def test_a_real_balance_reaches_the_view_as_exact_rao() -> None:
    """Stake must never round-trip through a float on the way in."""
    client = _Client(
        metagraph=_metagraph(
            hotkeys=["5Alice"], uids=[0], stake_tao=[123_456.789012345], permits=[False]
        )
    )
    assert adapter.sync_metagraph_view(client, netuid=42).stake_rao == (123_456_789_012_345,)


def test_an_unsynced_metagraph_does_not_read_as_a_deregistered_field() -> None:
    """The regression this file was rewritten to catch.

    A ``Metagraph`` carries an empty ``neurons`` list until it is synced with
    ``lite=False``. If that empty list wins over the populated parallel arrays,
    every miner reads as unregistered and a cycle burns a day of emission while
    logging ordinary churn.
    """
    raw = _metagraph(
        hotkeys=["5Alice", "5Bob"], uids=[4, 9], stake_tao=[1.0, 2.0], permits=[True, True]
    )
    assert raw.neurons == [], "precondition: an unsynced metagraph has no neuron rows"

    view = adapter.sync_metagraph_view(_Client(metagraph=raw), netuid=42)

    assert len(view) == 2
    assert view.is_registered("5Alice")


def test_an_actually_empty_metagraph_fails_the_cycle() -> None:
    """A live subnet always has its owner registered, so empty means broken."""
    empty = bittensor.Metagraph(netuid=42, sync=False, lite=False)
    with pytest.raises(SubtensorError, match="no neurons"):
        adapter.sync_metagraph_view(_Client(metagraph=empty), netuid=42)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


def test_the_hyperparameter_fields_the_gates_read_still_exist() -> None:
    names = {field.name for field in dataclasses.fields(bittensor.SubnetHyperparameters)}
    assert {"weights_version", "commit_reveal_weights_enabled"} <= names


def test_hyperparameters_read_through_the_adapter() -> None:
    fields = {field.name: None for field in dataclasses.fields(bittensor.SubnetHyperparameters)}
    fields.update(weights_version=7, commit_reveal_weights_enabled=True)
    raw = bittensor.SubnetHyperparameters(**fields)  # type: ignore[arg-type]

    result = adapter.read_hyperparameters(_Client(hyperparameters=raw), netuid=42)

    assert result.weights_version == 7
    assert result.commit_reveal_enabled is True


# ---------------------------------------------------------------------------
# Writes: the failure-reads-as-success traps
# ---------------------------------------------------------------------------


def test_an_extrinsic_response_is_unconditionally_truthy() -> None:
    """Why no code here may test the response for truth.

    ``ExtrinsicResponse`` defines no ``__bool__`` and no ``__len__``, so a
    *failed* response is still truthy. ``if result:`` reads every rejection as
    a success.
    """
    failed = _response(success=False, message="rejected")
    assert bool(failed) is True
    assert failed.success is False


def test_a_real_successful_submission_yields_a_lookupable_receipt() -> None:
    """The identifier must beat the status message.

    ``"Finalized."`` is not something an operator can look up. This pins that
    the adapter reaches the receipt's own identifier first.
    """
    client = _Client(
        set_weights=_response(
            success=True, message="Finalized.", extrinsic_receipt=_Receipt("1234-5")
        )
    )
    from prometheon.chain.weights import ChainWeightVector

    receipt = adapter.submit_set_weights(
        client,
        wallet=object(),
        netuid=42,
        vector=ChainWeightVector(uids=(4, 9), weights=(40_000, 25_535), hotkeys=("5A", "5B")),
        version_key=7,
        mechid=0,
    )
    assert receipt == "1234-5"


def test_a_receipt_that_cannot_identify_itself_does_not_fail_a_good_submission() -> None:
    """The chain accepted it; a missing diagnostic must not turn that into an error."""

    class Unresolved(_Receipt):
        def get_extrinsic_identifier(self) -> str | None:
            raise RuntimeError("block not yet available")

    client = _Client(
        set_weights=_response(success=True, message="Finalized.", extrinsic_receipt=Unresolved())
    )
    from prometheon.chain.weights import ChainWeightVector

    assert (
        adapter.submit_set_weights(
            client,
            wallet=object(),
            netuid=42,
            vector=ChainWeightVector(uids=(4,), weights=(65_535,), hotkeys=("5A",)),
            version_key=7,
            mechid=0,
        )
        == "Finalized."
    )


def test_a_real_failed_submission_raises_even_with_an_empty_message() -> None:
    """The trap: ``.message`` is often empty on failure, the detail is on ``.error``."""
    client = _Client(set_weights=_response(success=False, message="", error="RateLimit"))
    from prometheon.chain.weights import ChainWeightVector

    with pytest.raises(SetWeightsFailedError, match="RateLimit"):
        adapter.submit_set_weights(
            client,
            wallet=object(),
            netuid=42,
            vector=ChainWeightVector(uids=(4,), weights=(65_535,), hotkeys=("5A",)),
            version_key=7,
            mechid=0,
        )


def test_the_mechid_reaches_set_weights() -> None:
    """Weights on the wrong mechanism are accepted and then ignored."""
    client = _Client(set_weights=_response(success=True, message="ok"))
    from prometheon.chain.weights import ChainWeightVector

    adapter.submit_set_weights(
        client,
        wallet=object(),
        netuid=42,
        vector=ChainWeightVector(uids=(4,), weights=(65_535,), hotkeys=("5A",)),
        version_key=7,
        mechid=3,
    )
    assert client.calls[0][1]["mechid"] == 3


def test_set_weights_still_accepts_every_argument_the_adapter_passes() -> None:
    parameters = set(inspect.signature(bittensor.Subtensor.set_weights).parameters)
    assert {
        "wallet",
        "netuid",
        "uids",
        "weights",
        "version_key",
        "wait_for_inclusion",
        "mechid",
    } <= (parameters)


# ---------------------------------------------------------------------------
# Commitments
# ---------------------------------------------------------------------------


def test_set_commitment_takes_the_payload_as_a_string() -> None:
    """The codec emits ASCII text; an SDK that wanted bytes would silently differ."""
    parameters = inspect.signature(bittensor.Subtensor.set_commitment).parameters
    assert {"wallet", "netuid", "data"} <= set(parameters)
    assert "str" in str(parameters["data"].annotation)


def test_the_commitment_read_is_keyed_by_uid_on_this_sdk() -> None:
    """Recorded as a hazard rather than as a property worth having.

    A UID names a slot, and slots get recycled. The adapter therefore requires
    the caller to pass a UID resolved from the same snapshot the cycle is
    scoring. If a future SDK keys this by hotkey, this test fails and both the
    hazard and the ``uid`` argument can go away.
    """
    parameters = set(inspect.signature(bittensor.Subtensor.get_commitment).parameters)
    assert "uid" in parameters
    assert "hotkey_ss58" not in parameters


def test_the_commitment_block_is_readable_and_keyed_by_hotkey() -> None:
    """The block that settles a duplicate-model claim, taken from the authority.

    Duplicate claims on one revision SHA are resolved by ascending commit block.
    Nothing populated it for a while, so the sort collapsed to ascending uid —
    which hands every claim to whoever registered earliest, the attack
    ``registry.validation`` exists to prevent.

    The DB layer mirrors a block on its own commitment records and reading it
    from there would have been easier. It is read from chain instead, because
    duplicate resolution is an anti-gaming mechanism and sourcing it from the
    service being audited would make the anti-gaming property depend on the one
    component a validator is otherwise careful never to trust.

    Keyed by ``hotkey_ss58``, unlike ``get_commitment`` above — which is the
    better key, since a uid is a recycled slot.
    """
    parameters = set(inspect.signature(bittensor.Subtensor.get_commitment_metadata).parameters)
    assert "hotkey_ss58" in parameters
    assert "netuid" in parameters

    # The block has to survive as an integer field on the response, or
    # `read_commitment_block` silently falls back to "unknown" for everyone and
    # the ordering degrades to uid again without anything failing.
    from bittensor.core.types import CommitmentOfResponse

    assert "block" in typing.get_type_hints(CommitmentOfResponse)


def test_a_commitment_round_trips_through_the_sdk_boundary() -> None:
    commitment = ModelCommitment(hf_repo="acme/guard", hf_revision=REVISION)
    client = _Client(set_commitment=_response(success=True, message="ok"))

    payload = publish_commitment(client, wallet=object(), netuid=42, commitment=commitment)

    assert client.calls[0][1]["data"] == payload
    assert (
        read_commitment(_Client(commitment=payload), netuid=42, hotkey="5Alice", uid=4)
        == commitment
    )


def test_a_rejected_commitment_raises_rather_than_reporting_success() -> None:
    client = _Client(
        set_commitment=_response(success=False, message="", error="SpaceLimitExceeded")
    )
    with pytest.raises(CommitmentError, match="SpaceLimitExceeded"):
        publish_commitment(
            client,
            wallet=object(),
            netuid=42,
            commitment=ModelCommitment(hf_repo="acme/guard", hf_revision=REVISION),
        )
