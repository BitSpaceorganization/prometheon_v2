"""B10: a validator-side fault stops the cycle; a miner-side one costs the miner.

v2 turned every failure into a failed batch. A validator missing a package, on a
bad device string, or with a full disk scored every model zero and submitted,
handing the model pool to the burn hotkey for its own outage. Following the
install instructions in the docs was enough.

The evaluation runtime is an optional extra that CI does not install, so these
tests stand in for ``torch``, ``transformers`` and ``huggingface_hub`` with
small modules in ``sys.modules``. That covers the decisions this package makes
about their failures, not the libraries themselves.
"""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from typing import Any

import pytest

from prometheon.errors import ValidatorFaultError
from prometheon.evaluation.corpus import ItemSource, LabelledItem
from prometheon.evaluation.preflight import check_evaluation_runtime
from prometheon.evaluation.runner import (
    CheckpointUnavailableError,
    MinerTarget,
    download_checkpoint,
    evaluate_miner,
)

pytestmark = pytest.mark.unit

TARGET = MinerTarget(hotkey="5Miner", hf_repo="org/guard", hf_revision="a" * 40)


# --- stand-ins for the optional runtime ---------------------------------------


def _fake_torch(*, forward_fails: bool = False, nonfinite: bool = False) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = "f16", "bf16", "f32"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: False, empty_cache=lambda: None
    )
    torch.no_grad = nullcontext  # type: ignore[attr-defined]
    torch.tensor = lambda data, device=None: data  # type: ignore[attr-defined]

    class _Logits:
        def float(self) -> _Logits:
            return self

    def isfinite(_values: Any) -> Any:
        return types.SimpleNamespace(all=lambda: not nonfinite)

    torch.isfinite = isfinite  # type: ignore[attr-defined]
    torch._forward_fails = forward_fails  # type: ignore[attr-defined]
    torch._logits = _Logits()  # type: ignore[attr-defined]
    return torch


def _fake_transformers(torch: types.ModuleType) -> types.ModuleType:
    transformers = types.ModuleType("transformers")

    class Model:
        def to(self, _device: str) -> Model:
            return self

        def eval(self) -> None:
            return None

        def __call__(self, _ids: Any) -> Any:
            if torch._forward_fails:  # type: ignore[attr-defined]
                raise RuntimeError("CUDA error: no kernel image is available for execution")
            return types.SimpleNamespace(logits=torch._logits)  # type: ignore[attr-defined]

    transformers.GPT2Config = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    # `dtype`, the keyword transformers 5.x reads; the old `torch_dtype` alias
    # is refused here so the canary cannot drift back onto it.
    transformers.AutoModelForCausalLM = types.SimpleNamespace(  # type: ignore[attr-defined]
        from_config=lambda config, dtype=None: Model()
    )
    return transformers


def _install_runtime(monkeypatch: pytest.MonkeyPatch, **torch_options: bool) -> None:
    torch = _fake_torch(**torch_options)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", _fake_transformers(torch))


class TestThePreflight:
    def test_a_missing_runtime_stops_the_cycle_and_names_the_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(ValidatorFaultError, match="--extra wrapper"):
            check_evaluation_runtime(device="auto", dtype="float16")

    def test_a_healthy_runtime_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_runtime(monkeypatch)
        check_evaluation_runtime(device="cpu", dtype="float16")

    def test_a_device_that_cannot_run_anything_stops_the_cycle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The RTX 5090 on an old torch build: available, but every kernel fails."""
        _install_runtime(monkeypatch, forward_fails=True)
        with pytest.raises(ValidatorFaultError, match="no kernel image"):
            check_evaluation_runtime(device="cuda", dtype="float16")

    def test_nonfinite_output_stops_the_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_runtime(monkeypatch, nonfinite=True)
        with pytest.raises(ValidatorFaultError, match="non-finite"):
            check_evaluation_runtime(device="cpu", dtype="float16")

    def test_a_bad_dtype_is_a_configuration_fault(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_runtime(monkeypatch)
        with pytest.raises(ValidatorFaultError, match="misconfigured"):
            check_evaluation_runtime(device="cpu", dtype="float8")


class TestDownloads:
    def _hub(self, monkeypatch: pytest.MonkeyPatch, error: Exception | None) -> None:
        hub = types.ModuleType("huggingface_hub")

        def snapshot_download(repo: str, *, revision: str, allow_patterns: Any) -> str:
            if error is not None:
                raise error
            return f"/cache/{repo}/{revision}"

        hub.snapshot_download = snapshot_download  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def test_a_repository_the_miner_deleted_is_the_miners_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class RepositoryNotFoundError(Exception):
            pass

        self._hub(monkeypatch, RepositoryNotFoundError("404"))
        with pytest.raises(CheckpointUnavailableError):
            download_checkpoint(TARGET)

    def test_a_file_missing_from_the_repository_is_the_miners_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`huggingface_hub` 1.0 raises this as `RemoteEntryNotFoundError`.

        Errors are matched by name, so the rename alone would have made a
        miner's missing file a validator fault that stops the whole cycle.
        """

        class RemoteEntryNotFoundError(Exception):
            pass

        self._hub(monkeypatch, RemoteEntryNotFoundError("404"))
        with pytest.raises(CheckpointUnavailableError):
            download_checkpoint(TARGET)

    @pytest.mark.parametrize(
        "error",
        [OSError(28, "No space left on device"), ConnectionError("dns"), RuntimeError("429")],
    )
    def test_anything_else_is_this_validators_fault(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        self._hub(monkeypatch, error)
        with pytest.raises(ValidatorFaultError, match="failed on this validator"):
            download_checkpoint(TARGET)

    def test_a_missing_hub_library_is_this_validators_fault(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ValidatorFaultError, match="--extra wrapper"):
            download_checkpoint(TARGET)

    def test_a_good_download_returns_the_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._hub(monkeypatch, None)
        assert download_checkpoint(TARGET) == f"/cache/org/guard/{'a' * 40}"


class _Engine:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.unloaded = False

    def load(self) -> None:
        raise self.error

    def unload(self) -> None:
        self.unloaded = True

    def moderate(self, policy: str, items: Any) -> Any:  # pragma: no cover - never reached
        raise AssertionError("a model that did not load must not be asked anything")


ITEMS = [
    LabelledItem(item_id=f"p{index}", content="x", violating=False, source=ItemSource.PRODUCTION)
    for index in range(3)
]


class TestLoading:
    def test_a_missing_package_at_load_is_this_validators_fault(self) -> None:
        with pytest.raises(ValidatorFaultError, match="incomplete"):
            evaluate_miner(
                engine=_Engine(ImportError("accelerate")),  # type: ignore[arg-type]
                target=TARGET,
                items=ITEMS,
                policy="p",
                policy_version="v",
            )

    def test_a_checkpoint_that_will_not_load_is_scored_and_released(self) -> None:
        """S3: a failed load used to leave the weights on the GPU for the next miner."""
        engine = _Engine(RuntimeError("the checkpoint is corrupt"))
        result = evaluate_miner(
            engine=engine,  # type: ignore[arg-type]
            target=TARGET,
            items=ITEMS,
            policy="p",
            policy_version="v",
        )
        assert result.correct == 0
        assert result.failed_items == len(ITEMS)
        assert engine.unloaded
