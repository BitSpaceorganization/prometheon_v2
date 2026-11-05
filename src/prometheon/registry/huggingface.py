"""Independent verification of a miner's Hugging Face repository.

Answered here, from the public API and not from anything the miner or the hub
says: **does the repository at this exact commit contain only permitted files,
does it carry weights and a config, how large is it, and which weights are
they?** The last is the weights fingerprint, which is how duplicates are found:
by file content, not by git commit (v2's B7).

There used to be a second question -- whether ``miner.py`` was byte-for-byte
the canonical engine -- because miners served their own models and the engine
travelled in the repository. Validators run the engine themselves now, so there
is no miner-supplied code to hash and :data:`FILE_MANIFEST` stops being a
security boundary. It is a description instead: these are model files, and a
repository that is only ever read for weights has no business holding anything
else.

Revisions are always 40-character commit SHAs by the time they reach here. That
is what makes "verified once" mean anything: a branch verified today serves
different weights tomorrow.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

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
#: miner's repo any more, only its file list, but the cap stays on `fetch_file`
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
#: hide, and `deploy_config.yml`, a permitted file whose *contents* nothing
#: checked, turned out to be one of them.
#:
#: The engine is the validator's own now, so this list stops being a defence and
#: becomes a description: these are model files, and anything else does not
#: belong in a repository that is only ever read for weights.
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

#: The most weight files a repository may hold. Every file is one request for
#: every validator downloading it, so a repository split into thousands of tiny
#: shards could exhaust a validator's request budget on its own. Real
#: checkpoints of the permitted size use a few dozen at most.
MAX_WEIGHT_FILES: Final[int] = 256

_LFS_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


def file_manifest() -> frozenset[str]:
    """Every fixed filename a conforming repository may contain."""
    return FILE_MANIFEST


def file_allowed(name: str) -> bool:
    """Whether one repository entry is permitted, sharded weights included."""
    if name in FILE_MANIFEST:
        return True
    return name.startswith(SHARD_PREFIX) and name.endswith(SHARD_SUFFIX)


@dataclass(frozen=True)
class RepoFile:
    """One file at one commit, as the API describes it with ``?blobs=true``."""

    name: str
    #: Bytes, or ``None`` when the API reported no size.
    size: int | None
    #: The Git LFS object's sha256, or ``None`` for a file not stored in LFS.
    lfs_sha256: str | None


@dataclass(frozen=True)
class RepoSnapshot:
    """The file list of one repository at one commit."""

    repo: str
    revision: str
    files: tuple[str, ...]
    entries: tuple[RepoFile, ...] = ()

    @property
    def weight_files(self) -> tuple[RepoFile, ...]:
        return tuple(entry for entry in self.entries if entry.name.endswith(SHARD_SUFFIX))

    @property
    def weight_bytes(self) -> int:
        """Total size of the weights. Unknown sizes must be refused first."""
        return sum(entry.size or 0 for entry in self.weight_files)

    @property
    def non_weight_bytes(self) -> int:
        return sum(
            entry.size or 0 for entry in self.entries if not entry.name.endswith(SHARD_SUFFIX)
        )


def weights_fingerprint(snapshot: RepoSnapshot) -> str:
    """sha256 over the sorted LFS sha256s of every weight file, newline-joined.

    The same formula the hub uses at the freeze, recorded in the wire notes.
    File names do not count, so renaming a shard changes nothing; any change to
    any weight file's content changes it. A weight file not stored in LFS has
    no content hash the API will give, so it cannot be fingerprinted and the
    repository is refused.
    """
    digests: list[str] = []
    for entry in snapshot.weight_files:
        if entry.lfs_sha256 is None or not _LFS_SHA_RE.match(entry.lfs_sha256):
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision}: {entry.name} is not stored in Git LFS, so "
                "its content cannot be fingerprinted. Store weights with LFS"
            )
        digests.append(entry.lfs_sha256)
    if not digests:
        raise ManifestViolationError(
            f"{snapshot.repo}@{snapshot.revision} has no weights to fingerprint"
        )
    return hashlib.sha256("\n".join(sorted(digests)).encode("utf-8")).hexdigest()


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
        """Every file present at ``revision``, with sizes and LFS hashes.

        One request: ``?blobs=true`` is what makes the API report a size and an
        LFS object for each file, and every later check reads from this answer.
        """
        require_valid_repo_id(repo)
        require_valid_revision(revision)
        what = f"Hugging Face repo {repo}@{revision}"

        payload = self.get_json(f"/api/models/{repo}/revision/{revision}?blobs=true", what=what)

        siblings = payload.get("siblings")
        if not isinstance(siblings, list):
            raise RegistryError(f"{what} returned no file list")

        entries: list[RepoFile] = []
        for sibling in siblings:
            if not isinstance(sibling, dict):
                continue
            name = sibling.get("rfilename")
            if isinstance(name, str) and name:
                entries.append(_repo_file(name, sibling))
        if not entries:
            raise RegistryError(f"{what} lists no files")
        files = [entry.name for entry in entries]

        # The API echoes the commit it resolved. Anything other than the commit
        # asked for means the file list describes a different tree than the one
        # under verification.
        resolved = payload.get("sha")
        if isinstance(resolved, str) and _SHA_RE.match(resolved) and resolved != revision:
            raise RegistryError(f"{what} resolved to commit {resolved}")

        return RepoSnapshot(
            repo=repo,
            revision=revision,
            files=tuple(sorted(files)),
            entries=tuple(sorted(entries, key=lambda entry: entry.name)),
        )

    def weight_bytes(self, repo: str, revision: str) -> int:
        """Total size of the weight files at ``revision``, in bytes."""
        return self.list_files(repo, revision).weight_bytes

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
        """Raise unless every file is permitted and weights are present."""
        extra = sorted(name for name in snapshot.files if not file_allowed(name))
        if extra:
            shown = ", ".join(extra[:_MAX_REPORTED_FILES])
            if len(extra) > _MAX_REPORTED_FILES:
                shown += f", and {len(extra) - _MAX_REPORTED_FILES} more"
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} contains files outside the manifest: {shown}"
            )
        weights = [name for name in snapshot.files if name.endswith(SHARD_SUFFIX)]
        if not weights:
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} contains no {SHARD_SUFFIX} weights, "
                f"so there is no model to serve"
            )
        if len(weights) > MAX_WEIGHT_FILES:
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} splits its weights into {len(weights)} "
                f"files; at most {MAX_WEIGHT_FILES} are accepted"
            )
        # The engine cannot load a checkpoint without its config, and reading
        # the architecture needs it too. v2 let a repository without one pass
        # here and score zero at load instead (B17).
        if "config.json" not in snapshot.files:
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision} has no config.json, so it cannot be loaded"
            )
        unsized = [entry.name for entry in snapshot.entries if entry.size is None]
        if snapshot.entries and unsized:
            raise ManifestViolationError(
                f"{snapshot.repo}@{snapshot.revision}: Hugging Face reports no size for "
                f"{', '.join(unsized[:_MAX_REPORTED_FILES])}, so it cannot be checked against "
                "the size ceilings"
            )

    def verify_repository(self, repo: str, revision: str) -> RepoSnapshot:
        """Check the file list. Returns the snapshot that passed.

        One check now, where there used to be two. The engine hash is gone
        because the engine is no longer published here: the validator runs its
        own. Fetching a file from the miner's repo to verify code that ran in
        the miner's container was proving the wrong thing anyway.
        """
        snapshot = self.list_files(repo, revision)
        self.verify_manifest(snapshot)
        return snapshot


def _repo_file(name: str, sibling: dict[str, Any]) -> RepoFile:
    size = sibling.get("size")
    lfs = sibling.get("lfs")
    lfs_sha = lfs.get("sha256") if isinstance(lfs, dict) else None
    return RepoFile(
        name=name,
        size=size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else None,
        lfs_sha256=lfs_sha if isinstance(lfs_sha, str) else None,
    )


__all__ = [
    "DEFAULT_HF_ENDPOINT",
    "FILE_MANIFEST",
    "MAX_FILE_BYTES",
    "MAX_WEIGHT_FILES",
    "SHARD_PREFIX",
    "SHARD_SUFFIX",
    "HuggingFaceClient",
    "RepoFile",
    "RepoSnapshot",
    "file_allowed",
    "file_manifest",
    "is_valid_repo_id",
    "require_valid_repo_id",
    "weights_fingerprint",
]
