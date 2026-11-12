"""A model's parameter count, read from its safetensors headers without downloading it.

A ``.safetensors`` file starts with 8 bytes, a little-endian unsigned 64-bit
integer ``N``, followed by ``N`` bytes of JSON: one entry per tensor, each with
its ``dtype``, ``shape`` and ``data_offsets``, plus an optional
``__metadata__`` entry. The weights follow. So two small range requests per
file, the 8 bytes and then the header, give every tensor's shape, and an
oversized model is refused before a byte of its weights is fetched.

**The definition, which the hub uses too.** ``P`` is the sum, over every
``*.safetensors`` file in the repository at the frozen revision, of the
element count of every tensor, where a tensor's element count is the product
of its shape (1 for a scalar, whose shape is ``[]``). ``__metadata__`` is not a
tensor. Nothing else is read from the header, because nothing else is part of
the definition: two implementations that each check something extra would
disagree about some model, and then a validator and the hub would too.

**Every file counts.** A repository that ships the same weights twice, as one
``model.safetensors`` and as shards, is counted twice. The miner chose to
publish both.

A header this module cannot read makes the model invalid, as the miner's own
fault on every validator alike: one too long for the file, one past
:data:`MAX_HEADER_BYTES`, one that is not a JSON object of tensors with
non-negative integer shapes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, Final

from prometheon.errors import ManifestViolationError

#: Bytes of the length prefix at the start of every safetensors file.
LENGTH_PREFIX_BYTES: Final[int] = 8

#: The longest header read. A real header is one JSON entry per tensor, under
#: a hundred bytes each: a 4B-parameter model has a few hundred tensors, so its
#: header is tens of kilobytes. The cap bounds what one miner's file can make
#: every validator download and parse, and the hub applies the same one.
MAX_HEADER_BYTES: Final[int] = 8 * 1024 * 1024

#: The entry the format reserves for free-form string metadata.
_METADATA_KEY: Final[str] = "__metadata__"


def header_length(prefix: bytes, *, file_size: int, where: str) -> int:
    """The header length a file's first 8 bytes declare, checked against the file.

    ``file_size`` comes from the Hugging Face listing. A header that would run
    past the end of the file, or past :data:`MAX_HEADER_BYTES`, is refused
    rather than requested.
    """
    if len(prefix) != LENGTH_PREFIX_BYTES:
        raise ManifestViolationError(f"{where} is too short to be a safetensors file")
    length = int.from_bytes(prefix, "little", signed=False)
    if length == 0:
        raise ManifestViolationError(f"{where} declares an empty safetensors header")
    if length > MAX_HEADER_BYTES:
        raise ManifestViolationError(
            f"{where} declares a {length}-byte safetensors header, over the "
            f"{MAX_HEADER_BYTES}-byte limit"
        )
    if LENGTH_PREFIX_BYTES + length > file_size:
        raise ManifestViolationError(
            f"{where} declares a {length}-byte header in a {file_size}-byte file"
        )
    return length


def count_parameters(header: bytes, *, where: str) -> int:
    """The element count of every tensor one safetensors header describes."""
    try:
        parsed: Any = json.loads(header)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ManifestViolationError(f"{where} has a safetensors header that is not JSON") from exc
    if not isinstance(parsed, dict):
        raise ManifestViolationError(f"{where} has a safetensors header that is not a JSON object")
    total = 0
    for name, entry in parsed.items():
        if name == _METADATA_KEY:
            continue
        total += _elements(entry, where=where)
    return total


def _elements(entry: object, *, where: str) -> int:
    shape = entry.get("shape") if isinstance(entry, Mapping) else None
    if not isinstance(shape, list):
        raise ManifestViolationError(f"{where} has a tensor without a shape in its header")
    # bool first: `isinstance(True, int)` is true, and a dimension of `true`
    # is not a dimension.
    if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
        raise ManifestViolationError(
            f"{where} has a tensor whose shape is not a list of non-negative integers"
        )
    return math.prod(shape)


__all__ = [
    "LENGTH_PREFIX_BYTES",
    "MAX_HEADER_BYTES",
    "count_parameters",
    "header_length",
]
