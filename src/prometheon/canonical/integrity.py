"""Canonical artifact integrity: rendering, hashing, and verification.

Two artifacts are pinned, by different means because they differ in kind:

- **The engine** (``miner.py``) is byte-identical for every miner, so a plain
  SHA-256 settles it.
- **The wrapper** carries four per-miner values, so it is normalised first:
  mask the four assignments, parse, and re-emit the tree through this module's
  own serialiser. Comparing normalised forms means formatting, comments, and
  whitespace cannot change the verdict, while any change to a statement does.

The serialiser is hand-written rather than ``ast.dump`` because this hash is a
consensus value. ``ast.dump`` is a debugging repr with no stability guarantee,
and it moved twice inside the supported Python range, giving three different
hashes for one wrapper. See :func:`normalise_source`.

Hashing the raw text instead would let a miner invalidate their own deployment
by reformatting, and would let a reviewer believe two files differ when only a
comment moved.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Final

from prometheon.errors import RegistryError, RevisionFormatError

_ASSETS: Final[Path] = Path(__file__).resolve().parent / "assets"
_SCRIPT_ASSET: Final[Path] = _ASSETS / "miner.py.jinja2"

#: Everything a miner may set. Masked before hashing, so two miners running the
#: same script hash identically however they filled these in.
#:
#: The GPU values are here because a miner pays for their own hardware and a
#: model's size decides what it needs. They are integers assigned to module-level
#: constants — declarative configuration, not the arbitrary builder-method
#: reflection that let a miner run shell commands during an image build.
APPROVED_VARIABLES: Final[tuple[str, ...]] = (
    "PROMETHEON_HF_REPO",
    "PROMETHEON_HF_REVISION",
    "PROMETHEON_CHUTES_USER",
    "PROMETHEON_HOTKEY",
    "PROMETHEON_GPU_COUNT",
    "PROMETHEON_MIN_VRAM_GB",
)

#: Masked too, though a miner does not choose it: the image id is a subnet
#: constant substituted at render time. Masking it decouples the two
#: migrations — rebuilding the image (a torch bump, a CUDA bump) would otherwise
#: change every miner's script hash and force a script migration alongside it.
#: The deployed image is verified separately, against the chute record.
_MASKED_VARIABLES: Final[tuple[str, ...]] = (
    *APPROVED_VARIABLES,
    "PROMETHEON_IMAGE_ID",
)

_HF_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")


def chute_namespace_prefix() -> str:
    """The prefix every chute name and slug carries — ``prometheon`` by default.

    Overridable via ``PROMETHEON_CHUTE_PREFIX`` for testnet only: Chutes reserves
    the ``prometheon`` namespace to a specific mainnet netuid, so a testnet miner
    cannot deploy under it. A testnet run sets a non-reserved prefix here, and the
    validator's slug check reads the same env var, so the rendered wrapper and the
    canonical hash stay in lockstep on both sides. Mainnet leaves it unset and the
    behaviour — and the hash — is exactly ``prometheon``.
    """
    return os.environ.get("PROMETHEON_CHUTE_PREFIX", "prometheon")


def is_valid_revision(revision: str) -> bool:
    """True only for a 40-character lowercase hex commit SHA.

    Branches and tags are rejected: a moving reference is not a commitment, and
    accepting one lets a miner change the model under a revision that was
    already verified.
    """
    return bool(_HF_SHA_RE.match(revision or ""))


def require_valid_revision(revision: str) -> str:
    if not is_valid_revision(revision):
        raise RevisionFormatError(
            f"revision must be a 40-character lowercase hex SHA, got {revision!r}"
        )
    return revision


@lru_cache(maxsize=1)
def wrapper_template_source() -> str:
    """The canonical miner script, as shipped.

    One artifact. The engine used to be a second file published in the miner's
    own Hugging Face repo and hash-checked at load time by code running inside
    an image the miner had built — which proved which bytes were on disk and
    nothing about the runtime that loaded them. It is in here now, inside the
    region validators hash from the outside.
    """
    return _SCRIPT_ASSET.read_text(encoding="utf-8")


#: GPU envelope a miner may ask for, matching `chutes.chute.node_selector`.
#: Bounded here so a value the platform would refuse is caught while rendering
#: rather than on a failed deploy.
MIN_GPU_COUNT: Final[int] = 1
MAX_GPU_COUNT: Final[int] = 8
MIN_VRAM_GB: Final[int] = 16
MAX_VRAM_GB: Final[int] = 140


def render_wrapper(
    *,
    hf_repo: str,
    hf_revision: str,
    chutes_user: str,
    hotkey: str,
    image_id: str,
    gpu_count: int = MIN_GPU_COUNT,
    min_vram_gb: int = MIN_VRAM_GB,
) -> str:
    """Render the miner script a miner deploys.

    Rendered by plain substitution rather than by a template engine's expression
    language: the script must contain nothing a miner could influence beyond
    these values, and a full template engine is a larger surface than this needs.

    ``hotkey`` derives the chute's name, which is what makes its id computable —
    ``uuid5(NAMESPACE_OID, f"{username}::chute::{name}")`` — and therefore
    checkable against the commitment. It also binds one deployment to one
    identity.
    """
    require_valid_revision(hf_revision)
    for name, value in (
        ("hf_repo", hf_repo),
        ("chutes_user", chutes_user),
        ("hotkey", hotkey),
        ("image_id", image_id),
    ):
        if not value or '"' in value or "\n" in value:
            raise RegistryError(f"{name} must be a non-empty single-line value without quotes")

    if not MIN_GPU_COUNT <= gpu_count <= MAX_GPU_COUNT:
        raise RegistryError(
            f"gpu_count must be between {MIN_GPU_COUNT} and {MAX_GPU_COUNT}, got {gpu_count}"
        )
    if not MIN_VRAM_GB <= min_vram_gb <= MAX_VRAM_GB:
        raise RegistryError(
            f"min_vram_gb must be between {MIN_VRAM_GB} and {MAX_VRAM_GB}, got {min_vram_gb}"
        )

    source = wrapper_template_source()
    replacements = {
        "{{ hf_repo }}": hf_repo,
        "{{ hf_revision }}": hf_revision,
        "{{ chutes_user }}": chutes_user,
        "{{ hotkey }}": hotkey,
        "{{ image_id }}": image_id,
        "{{ gpu_count }}": str(gpu_count),
        "{{ min_vram_gb }}": str(min_vram_gb),
        "{{ chute_prefix }}": chute_namespace_prefix(),
    }
    for placeholder, value in replacements.items():
        source = source.replace(placeholder, value)

    if "{{" in source:
        raise RegistryError("the miner script still contains unsubstituted placeholders")
    return source


#: Node type -> the fields this package serialises, in this fixed order.
#:
#: Three kinds of field are excluded on purpose, and they are precisely the ones
#: that broke the hash across CPython releases:
#:
#: - ``type_params`` (added 3.12 to ``FunctionDef``/``AsyncFunctionDef``/``ClassDef``)
#: - ``type_comment`` and ``type_ignores`` (always ``None``/empty here, and a
#:   dumping change around defaults is what moved 3.13)
#: - ``kind`` on ``Constant`` (the ``u''`` prefix marker, always ``None``)
#:
#: The table is an allowlist rather than a denylist. A node type absent from it
#: is refused, which keeps the output stable against future syntax *and* means
#: a deployment using a construct the canonical wrapper never contained is
#: rejected before any hash comparison happens.
_CANONICAL_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    # modules and statements
    "Module": ("body",),
    "FunctionDef": ("name", "args", "body", "decorator_list", "returns"),
    "AsyncFunctionDef": ("name", "args", "body", "decorator_list", "returns"),
    "ClassDef": ("name", "bases", "keywords", "body", "decorator_list"),
    "Return": ("value",),
    "Assign": ("targets", "value"),
    "AnnAssign": ("target", "annotation", "value", "simple"),
    "AugAssign": ("target", "op", "value"),
    "For": ("target", "iter", "body", "orelse"),
    "AsyncFor": ("target", "iter", "body", "orelse"),
    "While": ("test", "body", "orelse"),
    "If": ("test", "body", "orelse"),
    "With": ("items", "body"),
    "AsyncWith": ("items", "body"),
    "Raise": ("exc", "cause"),
    "Try": ("body", "handlers", "orelse", "finalbody"),
    "Assert": ("test", "msg"),
    "Import": ("names",),
    "ImportFrom": ("module", "names", "level"),
    "Global": ("names",),
    "Nonlocal": ("names",),
    "Expr": ("value",),
    "Pass": (),
    "Break": (),
    "Continue": (),
    "ExceptHandler": ("type", "name", "body"),
    # expressions
    "BoolOp": ("op", "values"),
    "BinOp": ("left", "op", "right"),
    "UnaryOp": ("op", "operand"),
    "Lambda": ("args", "body"),
    "IfExp": ("test", "body", "orelse"),
    "Dict": ("keys", "values"),
    "Set": ("elts",),
    "ListComp": ("elt", "generators"),
    "SetComp": ("elt", "generators"),
    "DictComp": ("key", "value", "generators"),
    "GeneratorExp": ("elt", "generators"),
    "Await": ("value",),
    "Yield": ("value",),
    "YieldFrom": ("value",),
    "Compare": ("left", "ops", "comparators"),
    "Call": ("func", "args", "keywords"),
    "FormattedValue": ("value", "conversion", "format_spec"),
    "JoinedStr": ("values",),
    "Constant": ("value",),
    "Attribute": ("value", "attr", "ctx"),
    "Subscript": ("value", "slice", "ctx"),
    "Starred": ("value", "ctx"),
    "Name": ("id", "ctx"),
    "List": ("elts", "ctx"),
    "Tuple": ("elts", "ctx"),
    "Slice": ("lower", "upper", "step"),
    # helpers
    "comprehension": ("target", "iter", "ifs", "is_async"),
    "arguments": (
        "posonlyargs",
        "args",
        "vararg",
        "kwonlyargs",
        "kw_defaults",
        "kwarg",
        "defaults",
    ),
    "arg": ("arg", "annotation"),
    "keyword": ("arg", "value"),
    "alias": ("name", "asname"),
    "withitem": ("context_expr", "optional_vars"),
    # contexts and operators carry no fields; their type name is the value
    "Load": (),
    "Store": (),
    "Del": (),
    "And": (),
    "Or": (),
    "Add": (),
    "Sub": (),
    "Mult": (),
    "Div": (),
    "FloorDiv": (),
    "Mod": (),
    "Pow": (),
    "LShift": (),
    "RShift": (),
    "BitOr": (),
    "BitXor": (),
    "BitAnd": (),
    "MatMult": (),
    "Invert": (),
    "Not": (),
    "UAdd": (),
    "USub": (),
    "Eq": (),
    "NotEq": (),
    "Lt": (),
    "LtE": (),
    "Gt": (),
    "GtE": (),
    "Is": (),
    "IsNot": (),
    "In": (),
    "NotIn": (),
}


def _mask_variables(source: str) -> str:
    """Blank the per-miner assignments so only the fixed code is compared."""
    for name in _MASKED_VARIABLES:
        # String *and* integer literals: the GPU values are ints, and a mask
        # that only understood quotes would leave them in the hashed region and
        # give every GPU choice a different script hash.
        source = re.sub(
            rf'^{name}\s*=\s*(".*?"|\'.*?\'|-?\d[\d_]*)',
            f'{name} = ""',
            source,
            flags=re.MULTILINE | re.DOTALL,
        )
    return source


def normalise_source(source: str) -> str:
    """Mask the approved values, then reduce to a formatting-free canonical form.

    The output is stable across CPython releases, which is the entire reason
    this function is hand-written instead of one line of :func:`ast.dump`.

    **``ast.dump`` cannot be used here, and the reason is worth recording,
    because it produced a subnet-breaking defect that every test passed.**
    Its output is a debugging repr with no compatibility guarantee, and it has
    moved twice inside this package's supported range: 3.12 added
    ``type_params`` to function and class definitions, and 3.13 began omitting
    fields that equal their defaults (``keywords=[]``, ``decorator_list=[]``,
    ``posonlyargs=[]``, ``orelse=[]``). So the same wrapper source hashed to
    three different values:

        3.10  9d919270a8ee16f1...
        3.12  e9b87ac26740d14c...
        3.13  6c72d0453e63e52f...

    That hash is the subnet's central published pin. Two validators on
    different interpreters would compute different ``ACCEPTED_WRAPPER_HASHES``
    and reject each other's miners, so every miner would be eligible on some
    validators and invalid on others. ``LEGACY_WRAPPER_HASHES`` holds frozen
    hex literals, so the documented migration window would be wrong everywhere
    except the machine the literal was generated on.

    Nothing caught it: the contract tests compared ``wrapper_hash(...)``
    against ``canonical_wrapper_hash()``, both computed by the running
    interpreter, which proves self-consistency and not correctness. The fix is
    a serialisation this repository owns, plus a literal pin in
    ``tests/contract/test_shipped_artifacts.py`` checked on every Python in
    the CI matrix.

    Only the node types the wrapper grammar actually uses are emitted, and an
    unknown type is refused rather than rendered. That doubles as a security
    property: the canonical wrapper is a small fixed set of constructs, so a
    deployment containing anything else is not the canonical wrapper and does
    not need a hash comparison to say so.
    """
    masked = _mask_variables(source)
    try:
        tree = ast.parse(masked)
    except SyntaxError as exc:
        raise RegistryError(f"source is not valid Python: {exc}") from exc
    return _canonical_node(tree)


def _canonical_node(node: object) -> str:
    """One AST node as stable text, recursing over its children.

    Empty collections and ``None`` are omitted rather than rendered, so a
    release that starts or stops emitting a defaulted field cannot change the
    output. Fields are emitted in the order this module fixes, never in the
    order the running interpreter happens to list them.
    """
    if isinstance(node, ast.AST):
        name = type(node).__name__
        fields = _CANONICAL_FIELDS.get(name)
        if fields is None:
            raise RegistryError(
                f"source contains a {name!r} node, which the canonical wrapper "
                "grammar does not use; this is not the canonical wrapper"
            )
        parts = []
        for field in fields:
            value = getattr(node, field, None)
            if value is None or (isinstance(value, list) and not value):
                continue
            parts.append(f"{field}={_canonical_node(value)}")
        return f"{name}({','.join(parts)})"

    if isinstance(node, list):
        return "[" + ",".join(_canonical_node(item) for item in node) + "]"

    # Constants and identifiers. `repr` is stable for str/int/float/bool/None
    # and is what makes two spellings of the same string compare equal.
    return repr(node)


def wrapper_hash(source: str) -> str:
    """The comparable hash of a wrapper, rendered or deployed."""
    return hashlib.sha256(normalise_source(source).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def canonical_wrapper_hash() -> str:
    """The hash a conforming deployment must produce."""
    return wrapper_hash(
        render_wrapper(
            hf_repo="placeholder/repo",
            hf_revision="0" * 40,
            chutes_user="placeholder",
            hotkey="placeholder",
            image_id="placeholder",
        )
    )


def extract_declared_variables(source: str) -> dict[str, str]:
    """Read the four approved values out of a deployed wrapper.

    Walks module-level assignments rather than pattern-matching text. A regex
    would happily read a SHA out of ``# PROMETHEON_HF_REVISION = "..."`` and
    miss the real binding underneath it. That is how a miner would deploy
    against a moving reference while committing a clean SHA on chain.
    """
    found: dict[str, str] = dict.fromkeys(APPROVED_VARIABLES, "")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found

    wanted = set(APPROVED_VARIABLES)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            found[target.id] = node.value.value
    return found


__all__ = [
    "APPROVED_VARIABLES",
    "MAX_GPU_COUNT",
    "MAX_VRAM_GB",
    "MIN_GPU_COUNT",
    "MIN_VRAM_GB",
    "canonical_wrapper_hash",
    "extract_declared_variables",
    "is_valid_revision",
    "normalise_source",
    "render_wrapper",
    "require_valid_revision",
    "wrapper_hash",
    "wrapper_template_source",
]
