"""Independent verification of a miner's Hugging Face repository.

Two questions are answered here, both from the public API and neither from
anything the miner or the DB layer says:

1. Does the repository at this exact commit contain only permitted files?
2. Is ``miner.py`` byte-for-byte the canonical engine?

The manifest is *read out of the canonical wrapper template* rather than
restated in this module. A validator refuses to run a repository holding a file the
wrapper's manifest excludes; a validator carrying its own copy of that list
would, the moment the two drifted, either reject repositories that deploy fine
or accept repositories that cannot.

Revisions are always 40-character commit SHAs by the time they reach here. That
is what makes "verified once" mean anything: a branch verified today serves
different weights tomorrow.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx

from prometheon.errors import (
    ManifestViolationError,
    RegistryError,
)
from prometheon.registry._http import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    RegistryHttpClient,
)
from prometheon.revision import require_valid_revision
from prometheon.version import __version__

DEFAULT_HF_ENDPOINT: Final[str] = "https://huggingface.co"


_USER_AGENT: Final[str] = f"prometheon-validator/{__version__}"

#: Ceiling on any single file this client will read. Nothing is fetched from a
#: miner's repo any more — only its file list — but the cap stays on `fetch_file`
#: because the path is miner-controlled and an unbounded read is a denial of
#: service any miner could trigger.
MAX_FILE_BYTES: Final[int] = 1 << 20

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


#: Every file a conforming repository may contain.
#:
#: **Weights, config and tokenizer. No executable code of any kind.** The engine
#: used to be published here as `miner.py` and hash-checked at load time, which
#: made this list a security boundary: it had to enumerate the ways code could
#: hide, and `deploy_config.yml` — a permitted file whose *contents* nothing
#: checked — turned out to be one of them.
#:
#: The engine is in the deploy script now, inside the region validators hash. So
#: this list stops being a defence and becomes a description: these are model
#: files, and anything else does not belong in a repository that is only ever
#: read for weights.
FILE_MANIFEST: Final[frozenset[str]] = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "LICENSE",
        "LICENSE.txt",
        "README.md",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)

#: Sharded weights are the one legitimate variable-name case.
SHARD_PREFIX: Final[str] = "model-"
SHARD_SUFFIX: Final[str] = ".safetensors"


def file_manifest() -> frozenset[str]:
    """Every fixed filename a conforming repository may contain."""
    return FILE_MANIFEST


def file_allowed(name: str) -> bool:
    """Whether one repository entry is permitted, sharded weights included."""
    if name in FILE_MANIFEST:
        return True
    return name.startswith(SHARD_PREFIX) and name.endswith(SHARD_SUFFIX)


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

    def weight_bytes(self, repo: str, revision: str) -> int:
        """Total size of the weight shards at ``revision``, in bytes.

        Read from the API rather than by downloading, because the point is to
        refuse an oversized model *before* every validator on the subnet spends
        an hour pulling it. `?blobs=true` is what makes the API report sizes at
        all; without it `siblings` carries names only.

        Returns ``0`` when the API reports no sizes. A missing number is not
        evidence of a small model, so callers treat it as unknown rather than
        as a pass.
        """
        require_valid_repo_id(repo)
        require_valid_revision(revision)
        what = f"Hugging Face repo {repo}@{revision}"
        payload = self.get_json(f"/api/models/{repo}/revision/{revision}?blobs=true", what=what)
        siblings = payload.get("siblings")
        if not isinstance(siblings, list):
            return 0

        total = 0
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            name = sibling.get("rfilename")
            size = sibling.get("size")
            if (
                isinstance(name, str)
                and name.endswith(SHARD_SUFFIX)
                and isinstance(size, int)
                and size > 0
            ):
                total += size
        return total

    def fetch_file(
        self, repo: str, revision: str, filename: str, *, max_bytes: int = MAX_FILE_BYTES
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
        if not any(name.endswith(SHARD_SUFFIX) for name in snapshot.files):
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} contains no {SHARD_SUFFIX} weights, "
                f"so there is no model to serve"
            )

    def verify_repository(self, repo: str, revision: str) -> RepoSnapshot:
        """Check the file list. Returns the snapshot that passed.

        One check now, where there used to be two. The engine hash is gone
        because the engine is no longer published here: it lives in the deploy
        script, and the script's own hash covers it. Fetching a file from the
        miner's repo to verify code that runs in the miner's container was
        proving the wrong thing anyway.
        """
        snapshot = self.list_files(repo, revision)
        self.verify_manifest(snapshot)
        return snapshot


__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "FILE_MANIFEST",
    "MAX_FILE_BYTES",
    "SHARD_PREFIX",
    "SHARD_SUFFIX",
    "HuggingFaceClient",
    "RepoSnapshot",
    "file_allowed",
    "file_manifest",
    "is_valid_repo_id",
    "require_valid_repo_id",
]
