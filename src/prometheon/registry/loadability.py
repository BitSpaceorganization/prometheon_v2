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
from typing import Any, Final

from pathlib import Path

from prometheon.errors import RegistryError

#: `config.json` is small. A checkpoint shipping a huge one is not one this can
#: reason about anyway.
_MAX_CONFIG_BYTES: Final[int] = 1 << 20

#: The architectures the deployment image can load, frozen from its pinned
#: transformers. See the file's header for how to regenerate it.
_MODEL_TYPES_FILE: Final[Path] = (
    Path(__file__).parent / "assets" / "transformers-4.46-model-types.txt"
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
    """Every ``model_type`` the *image's* transformers recognises.

    Read from a frozen list rather than from the installed package, and that is
    the whole point. The image pins its own `transformers`, so the question is
    never "can this machine load the checkpoint" but "can the container". A
    miner running a newer release locally would see a newer architecture
    accepted here and still watch every instance die inside the container --
    which is precisely the failure this exists to prevent.

    Regenerate the list when the pin in `pyproject.toml` moves; the file says
    how.
    """
    text = _MODEL_TYPES_FILE.read_text(encoding="utf-8")
    return frozenset(
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    )


def require_loadable(client: Any, repo: str, revision: str) -> str:
    """Raise unless the image's transformers can load this checkpoint.

    Returns the ``model_type`` that was checked, or ``""`` when the repository
    declares none -- an absent field is not evidence of a bad model, and this
    check does not guess.
    """
    known = known_model_types()

    model_type = model_type_of(client, repo, revision)
    if not model_type or model_type in known:
        return model_type

    raise ArchitectureUnsupportedError(
        f"{repo}@{revision} declares model_type {model_type!r}, which the installed "
        "transformers does not recognise. The deployment image pins the same range, so "
        "this checkpoint cannot be loaded there: the chute would deploy, report itself "
        "verified, then die during startup and reschedule, with no failure reason on "
        "the API. Choose a checkpoint whose architecture that transformers knows."
    )
