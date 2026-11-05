"""Is this validator able to run a model at all?

A failure while running a miner's model has two possible owners. If the
checkpoint is broken, the miner owns it, every validator sees the same thing,
and scoring that model zero is fair. If this validator is broken, a missing
package, a bad device string, a driver that cannot run the card, a full disk,
then every model fails here and only here, and submitting the resulting vector
hands the model pool to the burn hotkey for a validator's outage. v2 did exactly
that, and following its own install instructions was enough to trigger it (B10).

This module tells the two apart without a miner's model in the loop: it builds
a tiny random model on the configured device and dtype and runs one forward
pass. That is checked before anything is labelled, and again whenever a model
fails every batch, so a GPU that dies half way through the field stops the
cycle rather than scoring everyone after it zero. A full disk shows up where it
happens, as a download failure the runner classifies as this validator's.
"""

from __future__ import annotations

from typing import Any

from prometheon.errors import ValidatorFaultError

#: What installs the evaluation runtime.
_INSTALL_HINT = (
    "install it with `uv sync --extra wrapper` and run the validator from that environment"
)


def check_evaluation_runtime(*, device: str, dtype: str) -> None:
    """Raise :class:`ValidatorFaultError` unless a model can run here.

    A full disk is not checked up front: models already in the cache need no
    space, so any fixed threshold would stop healthy validators. A download
    that runs out of space raises as a validator fault where it happens.
    """
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ValidatorFaultError(
            f"the evaluation runtime is not installed ({exc.name or exc}); {_INSTALL_HINT}"
        ) from exc

    from prometheon.evaluation.engine import EngineError, resolve_device, resolve_dtype

    try:
        torch_dtype = resolve_dtype(dtype)
        torch_device = resolve_device(device)
    except EngineError as exc:
        raise ValidatorFaultError(f"[evaluation] is misconfigured: {exc}") from exc

    _canary(torch, transformers, device=torch_device, dtype=torch_dtype)


def _canary(torch: Any, transformers: Any, *, device: str, dtype: Any) -> None:
    """One forward pass of a random two-layer model, built locally, no download."""
    try:
        config = transformers.GPT2Config(
            n_layer=2, n_head=2, n_embd=16, n_positions=32, vocab_size=64
        )
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor([[1, 2, 3, 4]], device=device)).logits
        if not bool(torch.isfinite(logits.float()).all()):
            raise RuntimeError("the forward pass produced non-finite values")
    except ValidatorFaultError:
        raise
    except Exception as exc:
        raise ValidatorFaultError(
            f"a tiny local model could not run on device={device!r} dtype={dtype}: "
            f"{type(exc).__name__}: {exc}. Every miner's model would fail here too, so the "
            "cycle stops instead of scoring them all zero"
        ) from exc
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["check_evaluation_runtime"]
