"""Write the frozen list of architectures the evaluation runtime can run.

``model commit``, ``model verify``, every validator and the hub read
``src/prometheon/registry/assets/transformers_model_types.txt`` to decide whether
a checkpoint's ``model_type`` can be evaluated at all. Run this, in an
environment with the pinned ``wrapper`` extra installed, whenever the
transformers pin moves, and send the hub its new copy:

    uv run --extra wrapper python tools/freeze_model_types.py

A type is listed when two things hold, both checked against the installed
transformers:

1. ``AutoModelForCausalLM`` maps it to a model class, so the engine can load it.
2. The cache its default config builds holds only plain and sliding-window
   attention layers, and its modeling code builds no other cache. The engine
   computes the policy prefix once and shares it read-only across items; a
   linear-attention, recurrent or indexed layer writes into its own state, so
   the engine refuses it at load, and listing it would let a miner pay to enter
   a model that scores nothing.

A type whose default config cannot be built is left out too: nothing here can
show that the engine runs it.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

ASSET = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "prometheon"
    / "registry"
    / "assets"
    / "transformers_model_types.txt"
)

HEADER = """\
# Every model_type the evaluation runtime can run: AutoModelForCausalLM loads it,
# and its cache holds only plain and sliding-window attention layers, which the
# engine can share across items. Frozen from transformers {version}, the release
# pyproject pins for the wrapper extra ({pin}). Regenerate when that pin moves,
# and give the hub the new copy:
#
#   uv run --extra wrapper python tools/freeze_model_types.py
#
# Frozen rather than read from the local install: a newer transformers on the
# miner's machine would accept an architecture the runtime cannot load, and the
# only symptom is every validator failing to load it on the same day.
"""


def runnable_types() -> tuple[list[str], dict[str, str]]:
    """The listable types, and why each other causal-LM type is left out."""
    import transformers
    from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    logging.disable(logging.WARNING)
    models = Path(transformers.__file__).parent / "models"
    shareable = (DynamicLayer, DynamicSlidingWindowLayer)
    listed: list[str] = []
    left_out: dict[str, str] = {}
    for model_type in sorted(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES):
        try:
            cache = DynamicCache(config=CONFIG_MAPPING[model_type]())
        except Exception as exc:
            left_out[model_type] = f"its default config cannot be built ({type(exc).__name__})"
            continue
        kinds = sorted(
            {type(layer).__name__ for layer in cache.layers if type(layer) not in shareable}
        )
        stem = model_type.replace("-", "_")
        source = models / stem / f"modeling_{stem}.py"
        if source.exists():
            built = re.findall(r"past_key_values = (\w+)\(", source.read_text(encoding="utf-8"))
            other = {name for name in built if name not in ("DynamicCache", "EncoderDecoderCache")}
            kinds += sorted(other)
        if kinds:
            left_out[model_type] = "its cache holds " + ", ".join(kinds)
        else:
            listed.append(model_type)
    return listed, left_out


def main() -> int:
    import transformers

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'"transformers(?P<pin>[^"]*)"', pyproject)
    pin = match.group("pin") if match else "unknown"
    listed, left_out = runnable_types()
    ASSET.write_text(
        HEADER.format(version=transformers.__version__, pin=pin) + "\n".join(listed) + "\n",
        encoding="utf-8",
    )
    print(f"{len(listed)} types written to {ASSET}")
    for model_type, why in sorted(left_out.items()):
        print(f"  left out {model_type}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
