"""Independent verification of a miner's Hugging Face repository.

Two questions are answered here, both from the public API and neither from
anything the miner or the DB layer says:

1. Does the repository at this exact commit contain only permitted files?
2. Is ``miner.py`` byte-for-byte the canonical engine?

The manifest is *read out of the canonical wrapper template* rather than
restated in this module. The deployed chute refuses to start on a file the
wrapper's manifest excludes; a validator carrying its own copy of that list
would, the moment the two drifted, either reject repositories that deploy fine
or accept repositories that cannot.

Revisions are always 40-character commit SHAs by the time they reach here. That
is what makes "verified once" mean anything: a branch verified today serves
different weights tomorrow.
"""

from __future__ import annotations

import ast
import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

import httpx

from prometheon.canonical.integrity import (
    canonical_engine_sha256,
    require_valid_revision,
    wrapper_template_source,
)
from prometheon.errors import (
    EngineHashMismatchError,
    ManifestViolationError,
    RegistryError,
)
from prometheon.registry._http import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    RegistryHttpClient,
)
from prometheon.version import __version__

DEFAULT_HF_ENDPOINT: Final[str] = "https://huggingface.co"

ENGINE_FILENAME: Final[str] = "miner.py"

#: The canonical engine is about ten kilobytes. The ceiling exists because the
#: path is miner-controlled, not because the file is expected to grow.
MAX_ENGINE_BYTES: Final[int] = 1 << 20

_USER_AGENT: Final[str] = f"prometheon-validator/{__version__}"

#: ``namespace/name``, or a bare canonical name for the models Hugging Face
#: hosts without a namespace. Every segment must start alphanumeric, which is
#: what makes ``..`` unrepresentable. This value is interpolated into a URL
#: path, so traversal is refused before a request is made rather than sanitised
#: afterwards. Kept in step with the commitment encoder's own repo rule; a repo
#: the chain accepts and the registry rejects is a miner with no recourse.
_REPO_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$"
)

#: Single path segment, for the same reason.
_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")

#: How many offending names an error message carries before it is truncated.
_MAX_REPORTED_FILES: Final[int] = 10


def is_valid_repo_id(repo: str) -> bool:
    return bool(_REPO_ID_RE.match(repo or ""))


def require_valid_repo_id(repo: str) -> str:
    if not is_valid_repo_id(repo):
        raise RegistryError(f"not a Hugging Face repo id of the form owner/name: {repo!r}")
    return repo


def _literal_value(node: ast.expr) -> object | None:
    """Evaluate a module-level constant, seeing through ``frozenset(...)``."""
    target = node
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("frozenset", "set")
        and len(node.args) == 1
        and not node.keywords
    ):
        target = node.args[0]
    try:
        value: object = ast.literal_eval(target)
    except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
        return None
    return value


@lru_cache(maxsize=1)
def _manifest_rules() -> tuple[frozenset[str], str, str]:
    """The file manifest and shard affixes, parsed from the wrapper template."""
    try:
        tree = ast.parse(wrapper_template_source())
    except SyntaxError as exc:  # pragma: no cover - the asset is tested elsewhere
        raise RegistryError(f"the canonical wrapper template is not valid Python: {exc}") from exc

    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        literal = _literal_value(node.value)
        if literal is not None:
            values[target.id] = literal

    manifest = values.get("PROMETHEON_FILE_MANIFEST")
    prefix = values.get("PROMETHEON_SHARD_PREFIX")
    suffix = values.get("PROMETHEON_SHARD_SUFFIX")
    if (
        not isinstance(manifest, (set, frozenset))
        or not manifest
        or not all(isinstance(name, str) for name in manifest)
        or not isinstance(prefix, str)
        or not isinstance(suffix, str)
    ):
        raise RegistryError(
            "the canonical wrapper template does not declare a readable file manifest"
        )
    return frozenset(str(name) for name in manifest), prefix, suffix


def file_manifest() -> frozenset[str]:
    """Every fixed filename a conforming repository may contain."""
    return _manifest_rules()[0]


def file_allowed(name: str) -> bool:
    """Mirror of the deployed wrapper's own check, including sharded weights."""
    manifest, prefix, suffix = _manifest_rules()
    if name in manifest:
        return True
    return name.startswith(prefix) and name.endswith(suffix)


@dataclass(frozen=True)
class RepoSnapshot:
    """The file list of one repository at one commit."""

    repo: str
    revision: str
    files: tuple[str, ...]


class HuggingFaceClient(RegistryHttpClient):
    """Reads the public Hugging Face API. Never writes, never authenticates as a miner."""

    def __init__(
        self,
        *,
        endpoint: str = DEFAULT_HF_ENDPOINT,
        token: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        headers = {"User-Agent": _USER_AGENT}
        if token:
            # A read token only raises the anonymous rate limit. Verification
            # must never depend on private access: a repository a validator can
            # read and the public cannot is not an open model.
            headers["Authorization"] = f"Bearer {token}"
        super().__init__(
            base_url=endpoint,
            headers=headers,
            timeout_seconds=timeout_seconds,
            transport=transport,
            retries=retries,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )

    # -- fetching ---------------------------------------------------------

    def list_files(self, repo: str, revision: str) -> RepoSnapshot:
        """Every file present at ``revision``, as the API reports it."""
        require_valid_repo_id(repo)
        require_valid_revision(revision)
        what = f"Hugging Face repo {repo}@{revision}"

        payload = self.get_json(f"/api/models/{repo}/revision/{revision}", what=what)

        siblings = payload.get("siblings")
        if not isinstance(siblings, list):
            raise RegistryError(f"{what} returned no file list")

        files: list[str] = []
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            name = sibling.get("rfilename")
            if isinstance(name, str) and name:
                files.append(name)
        if not files:
            raise RegistryError(f"{what} lists no files")

        # The API echoes the commit it resolved. Anything other than the commit
        # asked for means the file list describes a different tree than the one
        # under verification.
        resolved = payload.get("sha")
        if isinstance(resolved, str) and _SHA_RE.match(resolved) and resolved != revision:
            raise RegistryError(f"{what} resolved to commit {resolved}")

        return RepoSnapshot(repo=repo, revision=revision, files=tuple(sorted(files)))

    def fetch_file(
        self, repo: str, revision: str, filename: str, *, max_bytes: int = MAX_ENGINE_BYTES
    ) -> bytes:
        """Raw bytes of one file at one commit."""
        require_valid_repo_id(repo)
        require_valid_revision(revision)
        if not _FILENAME_RE.match(filename):
            raise RegistryError(f"not a fetchable repository filename: {filename!r}")
        what = f"{filename} in {repo}@{revision}"
        return self.get_bytes(
            f"/{repo}/resolve/{revision}/{filename}", what=what, max_bytes=max_bytes
        )

    # -- verification -----------------------------------------------------

    def verify_manifest(self, snapshot: RepoSnapshot) -> None:
        """Raise unless every file is permitted and the engine is present."""
        extra = sorted(name for name in snapshot.files if not file_allowed(name))
        if extra:
            shown = ", ".join(extra[:_MAX_REPORTED_FILES])
            if len(extra) > _MAX_REPORTED_FILES:
                shown += f", and {len(extra) - _MAX_REPORTED_FILES} more"
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} contains files outside the manifest: {shown}"
            )
        if ENGINE_FILENAME not in snapshot.files:
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} is missing {ENGINE_FILENAME}"
            )

    def verify_engine(self, repo: str, revision: str) -> str:
        """Raise unless ``miner.py`` hashes to the canonical engine. Returns the digest."""
        raw = self.fetch_file(repo, revision, ENGINE_FILENAME, max_bytes=MAX_ENGINE_BYTES)
        digest = hashlib.sha256(raw).hexdigest()
        expected = canonical_engine_sha256()
        if digest != expected:
            raise EngineHashMismatchError(
                f"{ENGINE_FILENAME} in {repo}@{revision} hashes to {digest}, "
                f"not the canonical engine {expected}"
            )
        return digest

    def verify_repository(self, repo: str, revision: str) -> RepoSnapshot:
        """Manifest first, then engine hash. Returns the snapshot that passed."""
        snapshot = self.list_files(repo, revision)
        self.verify_manifest(snapshot)
        self.verify_engine(repo, revision)
        return snapshot


__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "ENGINE_FILENAME",
    "MAX_ENGINE_BYTES",
    "HuggingFaceClient",
    "RepoSnapshot",
    "file_allowed",
    "file_manifest",
    "is_valid_repo_id",
    "require_valid_repo_id",
]
