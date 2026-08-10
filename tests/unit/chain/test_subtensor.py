"""Behaviour of the bittensor SDK adapter, driven entirely through fakes.

No network and no chain: every test substitutes an object with the shape the
real SDK presents. What they are really about is *normalisation*. A failure in
any of the return shapes the SDK has used over time must raise instead of
reading as success.
"""

from __future__ import annotations

from typing import Any

import pytest

from prometheon.chain import subtensor as adapter
from prometheon.chain.weights import ChainWeightVector
from prometheon.config import ChainNetwork
from prometheon.errors import SetWeightsFailedError, SubtensorError

pytestmark = pytest.mark.unit


class FakeHyperparameters:
    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class FakeSubtensor:
    """A client carrying every accessor the adapter's compatibility gate wants.

    ``connect`` now refuses a client that is missing any of them, so a fake
    implementing only the method under test would fail the gate rather than the
    assertion, and every test would pass or fail for the wrong reason.
    """

    def __init__(self, **behaviour: Any) -> None:
        self.behaviour = behaviour
        self.calls: list[dict[str, Any]] = []

    def _answer(self, name: str) -> Any:
        value = self.behaviour[name]
        if isinstance(value, Exception):
            raise value
        return value

    def metagraph(self, *, netuid: int, lite: bool = True) -> Any:
        self.calls.append({"metagraph": netuid, "lite": lite})
        return self._answer("metagraph")

    def get_subnet_hyperparameters(self, *, netuid: int) -> Any:
        return self._answer("hyperparameters")

    def get_subnet_owner_hotkey(self, *, netuid: int) -> Any:
        return self._answer("owner")

    def get_commitment(self, *, netuid: int, uid: int) -> Any:
        return self._answer("commitment")

    def get_commitment_metadata(self, *, netuid: int, hotkey_ss58: str) -> Any:
        return self._answer("commitment_metadata")

    def set_weights(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._answer("set_weights")


class CompatibleClient:
    """The narrowest object ``assert_sdk_compatible`` accepts."""

    def metagraph(self, *, netuid: int, lite: bool = True) -> Any: ...
    def get_subnet_hyperparameters(self, *, netuid: int) -> Any: ...
    def get_subnet_owner_hotkey(self, *, netuid: int) -> Any: ...
    def get_commitment(self, *, netuid: int, uid: int) -> Any: ...
    def get_commitment_metadata(self, *, netuid: int, hotkey_ss58: str) -> Any: ...
    def set_weights(self, **kwargs: Any) -> Any: ...


class FakeMetagraph:
    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            setattr(self, name, value)


class ExtrinsicReceipt:
    """The receipt surface this SDK generation presents.

    There is no ``extrinsic_hash`` attribute here. ``ExtrinsicReceipt`` exposes
    ``get_extrinsic_identifier()`` and ``extrinsic_idx``, and a fake that
    invented a hash field agreed with an adapter that could never find one.
    """

    def __init__(self, identifier: str | None = None, index: int | None = None) -> None:
        self._identifier = identifier
        self.extrinsic_idx = index

    def get_extrinsic_identifier(self) -> str | None:
        return self._identifier


class ExtrinsicResponse:
    """The dataclass-shaped response current SDKs return.

    Notably *not* a tuple, which is the trap that made an earlier version of
    this parser swallow real failures.
    """

    def __init__(
        self,
        *,
        success: bool,
        message: str = "",
        error: Any = None,
        extrinsic_receipt: Any = None,
    ) -> None:
        self.success = success
        self.message = message
        self.error = error
        self.extrinsic_receipt = extrinsic_receipt


class TestConnect:
    def test_a_missing_sdk_names_the_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing() -> Any:
            raise SubtensorError("the bittensor SDK is not installed; install ...")

        monkeypatch.setattr(adapter, "_bittensor", missing)
        with pytest.raises(SubtensorError, match="not installed"):
            adapter.connect(ChainNetwork.FINNEY)

    def test_the_enum_is_passed_as_its_wire_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class Module:
            @staticmethod
            def Subtensor(*, network: str) -> Any:  # noqa: N802 - mirrors the SDK
                seen["network"] = network
                return CompatibleClient()

        monkeypatch.setattr(adapter, "_bittensor", lambda: Module)
        assert isinstance(adapter.connect(ChainNetwork.TEST), CompatibleClient)
        assert seen["network"] == "test"

    def test_a_custom_endpoint_string_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        class Module:
            @staticmethod
            def Subtensor(*, network: str) -> Any:  # noqa: N802 - mirrors the SDK
                seen["network"] = network
                return CompatibleClient()

        monkeypatch.setattr(adapter, "_bittensor", lambda: Module)
        assert isinstance(adapter.connect("ws://127.0.0.1:9944"), CompatibleClient)
        assert seen["network"] == "ws://127.0.0.1:9944"

    def test_an_empty_network_is_rejected_before_dialling(self) -> None:
        with pytest.raises(SubtensorError, match="non-empty"):
            adapter.connect("   ")

    def test_a_dial_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Module:
            @staticmethod
            def Subtensor(*, network: str) -> None:  # noqa: N802 - mirrors the SDK
                raise ConnectionRefusedError("no route to host")

        monkeypatch.setattr(adapter, "_bittensor", lambda: Module)
        with pytest.raises(SubtensorError, match="could not connect"):
            adapter.connect(ChainNetwork.LOCAL)


class TestDetectCapabilities:
    def test_mechid_is_detected_on_the_signature(self) -> None:
        class Modern:
            def set_weights(self, *, wallet: Any, netuid: int, mechid: int = 0) -> bool:
                return True

        assert adapter.detect_capabilities(Modern()).supports_mechid is True

    def test_an_sdk_without_mechid_reports_false(self) -> None:
        class Legacy:
            def set_weights(self, *, wallet: Any, netuid: int) -> bool:
                return True

        assert adapter.detect_capabilities(Legacy()).supports_mechid is False

    def test_an_uninspectable_attribute_fails_closed(self) -> None:
        """Knowing nothing must read as 'no support', never as 'supported'."""

        class Opaque:
            # Present but not introspectable: the shape a C extension or an
            # exotic proxy presents. inspect.signature refuses it.
            set_weights = object()

        assert adapter.detect_capabilities(Opaque()).supports_mechid is False

    def test_an_object_with_no_way_to_set_weights_is_an_incompatible_sdk(self) -> None:
        with pytest.raises(SubtensorError, match=r"no Subtensor\.set_weights"):
            adapter.detect_capabilities(object())

    def test_the_installed_class_is_inspected_when_nothing_is_passed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Module:
            class Subtensor:
                def set_weights(self, *, mechid: int = 0) -> bool:
                    return True

            __version__ = "10.5.0"

        monkeypatch.setattr(adapter, "_bittensor", lambda: Module)
        monkeypatch.setattr(adapter, "_sdk_version", lambda: "10.5.0")
        capabilities = adapter.detect_capabilities()
        assert capabilities.supports_mechid is True
        assert capabilities.sdk_version == "10.5.0"

    def test_an_sdk_without_a_subtensor_class_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(adapter, "_bittensor", lambda: object())
        with pytest.raises(SubtensorError, match="no Subtensor class"):
            adapter.detect_capabilities()


class TestSyncMetagraphView:
    def test_a_synced_view_carries_the_netuid_it_was_asked_for(self) -> None:
        subtensor = FakeSubtensor(
            metagraph=FakeMetagraph(
                hotkeys=["5Alice", "5Bob"],
                uids=[4, 9],
                total_stake=[1.0, 2.0],
                validator_permit=[True, False],
                block=555,
            )
        )
        snapshot = adapter.sync_metagraph_view(subtensor, netuid=42)
        assert snapshot.netuid == 42
        assert snapshot.uid_for("5Bob") == 9
        assert snapshot.block == 555

    def test_a_sync_failure_is_wrapped(self) -> None:
        subtensor = FakeSubtensor(metagraph=TimeoutError("substrate did not answer"))
        with pytest.raises(SubtensorError, match="could not sync the metagraph"):
            adapter.sync_metagraph_view(subtensor, netuid=42)

    def test_a_missing_subnet_is_named_as_such(self) -> None:
        with pytest.raises(SubtensorError, match="may not exist"):
            adapter.sync_metagraph_view(FakeSubtensor(metagraph=None), netuid=42)


class TestReadHyperparameters:
    def test_reads_both_gate_inputs_from_one_call(self) -> None:
        subtensor = FakeSubtensor(
            hyperparameters=FakeHyperparameters(
                commit_reveal_weights_enabled=True, weights_version=7
            )
        )
        result = adapter.read_hyperparameters(subtensor, netuid=42)
        assert result.commit_reveal_enabled is True
        assert result.weights_version == 7

    def test_an_unset_weights_version_collapses_to_the_documented_default(self) -> None:
        subtensor = FakeSubtensor(
            hyperparameters=FakeHyperparameters(
                commit_reveal_weights_enabled=False, weights_version=None
            )
        )
        assert adapter.read_hyperparameters(subtensor, netuid=42).weights_version == 0

    @pytest.mark.parametrize(
        "fields",
        [
            {"weights_version": 7},
            {"commit_reveal_weights_enabled": False},
        ],
    )
    def test_a_missing_field_fails_loud_instead_of_defaulting(self, fields: dict[str, Any]) -> None:
        """Defaulting commit-reveal to False once hid a live misconfiguration."""
        subtensor = FakeSubtensor(hyperparameters=FakeHyperparameters(**fields))
        with pytest.raises(SubtensorError, match="not compatible with this runtime"):
            adapter.read_hyperparameters(subtensor, netuid=42)

    def test_a_read_failure_is_wrapped(self) -> None:
        subtensor = FakeSubtensor(hyperparameters=ConnectionError("socket closed"))
        with pytest.raises(SubtensorError, match="could not read subnet hyperparameters"):
            adapter.read_hyperparameters(subtensor, netuid=42)

    def test_a_missing_subnet_is_named_as_such(self) -> None:
        with pytest.raises(SubtensorError, match="may not exist"):
            adapter.read_hyperparameters(FakeSubtensor(hyperparameters=None), netuid=42)


class TestReadSubnetOwnerHotkey:
    def test_reads_the_owner(self) -> None:
        class Owner:
            @staticmethod
            def get_subnet_owner_hotkey(*, netuid: int) -> str:
                return "5Owner"

        assert adapter.read_subnet_owner_hotkey(Owner(), netuid=42) == "5Owner"

    def test_a_missing_accessor_fails_loud_rather_than_defaulting(self) -> None:
        """A silently defaulted burn target would misroute real emission.

        Belt and braces: ``assert_sdk_compatible`` already refuses a client
        without this accessor at connect time, so reaching here means something
        bypassed ``connect``. It still has to raise rather than return a
        default. A burn routed to the wrong hotkey is real money.
        """
        with pytest.raises(SubtensorError, match="could not read the subnet owner hotkey"):
            adapter.read_subnet_owner_hotkey(object(), netuid=42)

    @pytest.mark.parametrize("value", [None, "", 0])
    def test_an_unusable_answer_is_rejected(self, value: Any) -> None:
        class Owner:
            @staticmethod
            def get_subnet_owner_hotkey(*, netuid: int) -> Any:
                return value

        with pytest.raises(SubtensorError, match="may not exist"):
            adapter.read_subnet_owner_hotkey(Owner(), netuid=42)

    def test_a_read_failure_is_wrapped(self) -> None:
        class Owner:
            @staticmethod
            def get_subnet_owner_hotkey(*, netuid: int) -> str:
                raise TimeoutError("no answer")

        with pytest.raises(SubtensorError, match="could not read the subnet owner hotkey"):
            adapter.read_subnet_owner_hotkey(Owner(), netuid=42)


VECTOR = ChainWeightVector(uids=(0, 1), weights=(32768, 32767), hotkeys=("5Alice", "5Bob"))


class TestSubmitSetWeights:
    def submit(self, result: Any, *, mechid: int | None = 0) -> tuple[str | None, FakeSubtensor]:
        subtensor = FakeSubtensor(set_weights=result)
        receipt = adapter.submit_set_weights(
            subtensor,
            wallet=object(),
            netuid=42,
            vector=VECTOR,
            version_key=7,
            mechid=mechid,
        )
        return receipt, subtensor

    def test_the_vector_is_passed_as_lists_the_sdk_can_consume(self) -> None:
        _, subtensor = self.submit(ExtrinsicResponse(success=True, message="Finalized."))
        call = subtensor.calls[0]
        assert call["uids"] == [0, 1]
        assert call["weights"] == [32768, 32767]
        assert call["version_key"] == 7
        assert call["wait_for_inclusion"] is True

    def test_mechid_is_passed_when_the_caller_supplies_one(self) -> None:
        _, subtensor = self.submit(ExtrinsicResponse(success=True), mechid=0)
        assert subtensor.calls[0]["mechid"] == 0

    def test_mechid_is_omitted_entirely_when_none(self) -> None:
        """Passing mechid=None to a legacy SDK is a TypeError, not a no-op."""
        _, subtensor = self.submit(ExtrinsicResponse(success=True), mechid=None)
        assert "mechid" not in subtensor.calls[0]

    def test_the_extrinsic_identifier_is_preferred_as_the_receipt(self) -> None:
        """ "Finalized." is not something an operator can look up; an id is."""
        receipt, _ = self.submit(
            ExtrinsicResponse(
                success=True,
                message="Finalized.",
                extrinsic_receipt=ExtrinsicReceipt("1234-5"),
            )
        )
        assert receipt == "1234-5"

    def test_the_extrinsic_index_is_the_next_best_receipt(self) -> None:
        receipt, _ = self.submit(
            ExtrinsicResponse(
                success=True, message="Finalized.", extrinsic_receipt=ExtrinsicReceipt(index=7)
            )
        )
        assert receipt == "extrinsic_idx=7"

    def test_a_receipt_that_raises_does_not_fail_an_accepted_submission(self) -> None:
        """The chain took the weights; a broken diagnostic must not undo that."""

        class Unresolved(ExtrinsicReceipt):
            def get_extrinsic_identifier(self) -> str | None:
                raise RuntimeError("block not yet available")

        receipt, _ = self.submit(
            ExtrinsicResponse(success=True, message="Finalized.", extrinsic_receipt=Unresolved())
        )
        assert receipt == "Finalized."

    def test_the_status_message_is_the_fallback_receipt(self) -> None:
        receipt, _ = self.submit(ExtrinsicResponse(success=True, message="Finalized."))
        assert receipt == "Finalized."

    def test_a_dataclass_failure_raises_rather_than_reading_as_success(self) -> None:
        """The response is tuple-ish but not a tuple; that is how failures hid."""
        with pytest.raises(SetWeightsFailedError, match="reported failure"):
            self.submit(ExtrinsicResponse(success=False, message="rate limit exceeded"))

    def test_a_failure_with_no_message_still_reports_every_diagnostic(self) -> None:
        with pytest.raises(SetWeightsFailedError) as excinfo:
            self.submit(ExtrinsicResponse(success=False, error="SettingWeightsTooFast"))
        assert "SettingWeightsTooFast" in str(excinfo.value)

    def test_a_failure_with_nothing_at_all_says_so(self) -> None:
        with pytest.raises(SetWeightsFailedError, match="no detail"):
            self.submit(ExtrinsicResponse(success=False))

    def test_a_long_receipt_is_truncated_to_stay_greppable(self) -> None:
        with pytest.raises(SetWeightsFailedError) as excinfo:
            self.submit(ExtrinsicResponse(success=False, extrinsic_receipt="x" * 5000))
        assert len(str(excinfo.value)) < 500
        assert "..." in str(excinfo.value)

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            ((True, "ok"), "ok"),
            ((True, ""), None),
            (True, None),
            ("0xabc123", "0xabc123"),
            (None, None),
        ],
    )
    def test_legacy_return_shapes_are_normalised(self, result: Any, expected: Any) -> None:
        receipt, _ = self.submit(result)
        assert receipt == expected

    def test_an_uninterpretable_return_refuses_to_read_as_success(self) -> None:
        """An unclassifiable return must not pass for success.

        An object the parser cannot classify used to return "no receipt" and
        the cycle carried on believing the weights had landed. There is no safe
        way to guess, so it raises.
        """
        with pytest.raises(SetWeightsFailedError, match=r"cannot.*interpret|refusing"):
            self.submit(object())

    @pytest.mark.parametrize("result", [(False, "rejected"), False])
    def test_legacy_failure_shapes_still_raise(self, result: Any) -> None:
        with pytest.raises(SetWeightsFailedError):
            self.submit(result)

    def test_an_sdk_exception_is_wrapped(self) -> None:
        with pytest.raises(SetWeightsFailedError, match="set_weights failed on netuid=42"):
            self.submit(ConnectionError("substrate closed the connection"))


class TestSdkCompatibilityGate:
    """The gate that turns a wrong pin into one startup error.

    An SDK generation mismatch has to be loud and immediate. The alternative
    happened here once: a validator ran a full cycle and burned a day of
    emission before anyone noticed.
    """

    def test_a_complete_client_passes(self) -> None:
        adapter.assert_sdk_compatible(CompatibleClient())

    def test_every_missing_accessor_is_named_at_once(self) -> None:
        """One error listing all of them, not five deploys learning one each."""

        class Partial:
            def metagraph(self, *, netuid: int, lite: bool = True) -> Any: ...

        with pytest.raises(SubtensorError) as excinfo:
            adapter.assert_sdk_compatible(Partial())

        message = str(excinfo.value)
        for name in (
            "get_subnet_hyperparameters",
            "get_subnet_owner_hotkey",
            "get_commitment",
            "get_commitment_metadata",
            "set_weights",
        ):
            assert name in message
        assert "bittensor>=10.5,<11" in message

    def test_a_read_intent_client_is_diagnosed_not_just_rejected(self) -> None:
        """The 11.x mis-pin, named as what it is.

        An operator who installs 11.x sees a client missing five methods. Left
        at that, the obvious reading is "upgrade further", which is the wrong
        direction. The message has to say the generation is different.
        """

        class ReadIntentClient:
            def read(self, name: str, **params: Any) -> Any: ...
            def execute(self, intent: Any, wallet: Any) -> Any: ...

        with pytest.raises(SubtensorError) as excinfo:
            adapter.assert_sdk_compatible(ReadIntentClient())

        message = str(excinfo.value)
        assert "11.x" in message
        assert "commitment" in message

    def test_a_non_callable_of_the_right_name_does_not_satisfy_the_gate(self) -> None:
        """A same-named attribute does not satisfy the gate; only a callable does."""

        class Decoy(CompatibleClient):
            set_weights = "not callable"  # type: ignore[assignment]

        with pytest.raises(SubtensorError, match="set_weights"):
            adapter.assert_sdk_compatible(Decoy())
