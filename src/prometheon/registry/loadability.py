"""Can the image's transformers actually load this checkpoint?

The image pins a `transformers` range, and a checkpoint whose architecture that
release does not know cannot be loaded at all -- however good the model is. That
failure happens inside the container, at `AutoModelForCausalLM.from_pretrained`,
long after everything a miner can see has succeeded::

    KeyError: 'qwen3'
    ValueError: The checkpoint you are trying to load has model type `qwen3`
    but Transformers does not recognize this architecture.

From outside, the commit is accepted, the deploy succeeds, an instance is
assigned and reports `verified: true`, and then it dies during startup and is
rescheduled. Repeatedly, with no failure reason anywhere on the API. That reads
as missing GPU capacity, and telling the two apart costs hours.

The check is cheap and local: read `model_type` out of the repository's
`config.json` and ask the installed `transformers` whether it knows it. It runs
where the miner still has a choice -- before rendering a wrapper, before
committing on chain, before paying for a deploy.
"""

from __future__ import annotations

import json
from typing import Any, Final

from prometheon.errors import RegistryError

#: `config.json` is small. A checkpoint shipping a huge one is not one this
#: check could reason about anyway.
_MAX_CONFIG_BYTES: Final[int] = 1 << 20


#: Distinguishes "caller passed nothing" from "caller passed None deliberately".
_UNSET: Final[object] = object()


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


def known_model_types() -> frozenset[str] | None:
    """Every ``model_type`` the installed transformers recognises.

    ``None`` when transformers is not installed, which is the common case on a
    machine that only drives the CLI. Callers report that as *unchecked* rather
    than as a pass: claiming a checkpoint loads when nothing verified it is how
    this failure stayed invisible in the first place.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    except Exception:  # pragma: no cover - depends on the optional `wrapper` extra
        return None
    return frozenset(CONFIG_MAPPING_NAMES)


def require_loadable(
    client: Any, repo: str, revision: str, *, known: frozenset[str] | object | None = _UNSET
) -> str | None:
    """Raise unless the installed transformers can load this checkpoint.

    Returns the ``model_type`` that was checked, or ``None`` when transformers
    is absent and nothing could be checked.

    ``known`` exists so the refusal itself can be tested without transformers
    installed. The rule this repository keeps is that a boundary test which
    skips is not a boundary test, and this boundary is the difference between a
    clear refusal and eleven silent instance deaths.
    """
    if known is _UNSET:
        known = known_model_types()
    if known is None:
        return None
    assert isinstance(known, frozenset)

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
