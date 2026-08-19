"""Can the image's transformers actually load this checkpoint?

The image pins a `transformers` range, and a checkpoint whose architecture that
release does not know cannot be loaded at all -- however good the model is. The
failure happens inside the container, at `AutoModelForCausalLM.from_pretrained`,
long after everything a miner can see has succeeded::

    KeyError: 'qwen3'
    ValueError: The checkpoint you are trying to load has model type `qwen3`
    but Transformers does not recognize this architecture.

From outside, the commit is accepted, the deploy succeeds, an instance is
assigned and reports `verified: true`, and then it dies during startup and is
rescheduled. Repeatedly, with no failure reason on the API. That reads as
missing GPU capacity, and telling the two apart cost hours.

The check is cheap and local: read `model_type` out of the repository's
`config.json` and ask the installed `transformers` whether it knows it. It runs
while the miner still has a choice -- before rendering a wrapper, before
committing on chain, before paying for a deploy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

from prometheon.errors import RegistryError

#: `config.json` is small. A checkpoint shipping a huge one is not one this can
#: reason about anyway.
_MAX_CONFIG_BYTES: Final[int] = 1 << 20

#: The architectures the image's pinned transformers knows, captured from it.
_MODEL_TYPES_ASSET: Final[Path] = (
    Path(__file__).resolve().parent / "assets" / "transformers_model_types.txt"
)


class ArchitectureUnsupportedError(RegistryError):
    """The pinned transformers cannot load this checkpoint's architecture."""

    code = "registry.architecture_unsupported"


def model_type_of(client: Any, repo: str, revision: str) -> str:
    """The ``model_type`` a repository declares, or ``""`` if it declares none."""
    try:
        raw = client.fetch_file(repo, revision, "config.json", max_bytes=_MAX_CONFIG_BYTES)
    except RegistryError:
        # A repository with no readable config.json fails the manifest check
        # anyway, and that check reports it better than this one would.
        return ""
    try:
        config = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    model_type = config.get("model_type") if isinstance(config, dict) else None
    return model_type if isinstance(model_type, str) else ""


def known_model_types() -> frozenset[str]:
    """Every ``model_type`` the deployment image's transformers recognises.

    Read from a frozen list, **not** from the transformers installed here. A
    miner's machine may carry a newer release that recognises an architecture
    the image cannot load; trusting it would return a pass for a checkpoint
    that then dies in the container, which is the exact failure this module
    exists to prevent. The image's pin is the authority, so the list is
    captured from it and shipped alongside the code.
    """
    return _frozen_model_types()


@lru_cache(maxsize=1)
def _frozen_model_types() -> frozenset[str]:
    text = _MODEL_TYPES_ASSET.read_text(encoding="utf-8")
    return frozenset(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def require_loadable(client: Any, repo: str, revision: str) -> str | None:
    """Raise unless the image's transformers can load this checkpoint.

    Returns the ``model_type`` that was checked, or ``""`` when the repository
    declares none -- an absent field is not evidence of a bad model, and this
    check does not guess.

    Never returns ``None``: the list is frozen and shipped, so there is always
    something to check against. It once could, back when the answer came from
    whatever transformers happened to be installed locally, and callers still
    have to be read with that in mind.
    """
    known = known_model_types()
    model_type = model_type_of(client, repo, revision)
    if not model_type or model_type in known:
        return model_type

    raise ArchitectureUnsupportedError(
        f"{repo}@{revision} declares model_type {model_type!r}, which "
        "the deployment image's transformers does not recognise. That image cannot load "
        "this checkpoint at all: every validator would fail to load it, "
        "on the same day, and the model would score zero everywhere. Choose a "
        "checkpoint whose architecture that runtime can load."
    )
