"""Independent verification of a miner's deployed chute.

Two facts are read from the platform: whether the chute is hot, and the source
that was actually deployed. Everything else a validator needs, which repo and
which commit the deployment serves, is read out of that source and never out of
the platform's metadata.

Chute metadata does not reliably expose the revision a deployment pinned, so a
validator that trusted it would let a miner commit a clean SHA on chain while
deploying against a branch, then swap the weights underneath a verification that
already passed. The deployed source is the artifact the wrapper hash covers, so
it is the only statement of intent that cannot be edited after the fact.

The chute slug is miner-controlled and becomes a hostname, so it is validated
against a DNS-label shape before any URL is built from it.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx

from prometheon.errors import ChuteNotRunningError, RegistryError
from prometheon.registry._http import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    RegistryHttpClient,
)

DEFAULT_CHUTES_API_BASE: Final[str] = "https://api.chutes.ai"

#: A deployed chute is reachable at its own subdomain.
DEFAULT_ENDPOINT_TEMPLATE: Final[str] = "https://{slug}.chutes.ai"

MODERATE_PATH: Final[str] = "/moderate"
HEALTH_PATH: Final[str] = "/health"

#: The wrapper source is a few hundred lines. The ceiling exists because the
#: body is miner-controlled.
MAX_SOURCE_BYTES: Final[int] = 1 << 20

#: A single DNS label: what may be interpolated into a hostname.
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")

#: Every chute this subnet evaluates lives under one namespace, agreed with the
#: Chutes team. The prefix is what makes a deployment provably ours: a slug
#: becomes a hostname, and without it a miner could commit any chute on the
#: platform — including someone else's working model — and be scored on it.
CHUTE_SLUG_PREFIX: Final[str] = "prometheon"

#: Prefix, then a separator, then at least one more character. A slug that is
#: exactly the prefix addresses nothing.
_PREFIXED_SLUG_RE: Final[re.Pattern[str]] = re.compile(
    rf"^{CHUTE_SLUG_PREFIX}-[a-z0-9](?:[a-z0-9-]{{0,{61 - len(CHUTE_SLUG_PREFIX) - 1}}}[a-z0-9])?$"
)

#: Chute ids are opaque platform identifiers interpolated into a URL path. The
#: character class matches what the commitment encoder accepts, so no id the
#: chain will carry is rejected here; ``.`` and ``..`` are excluded separately
#: because they are the two spellings that would traverse the path.
_CHUTE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_PATH_TRAVERSAL: Final[frozenset[str]] = frozenset({".", ".."})

#: Keys a code endpoint might wrap the source in, in preference order.
_SOURCE_KEYS: Final[tuple[str, ...]] = ("code", "source", "content")


def is_valid_chute_id(chute_id: str) -> bool:
    return bool(_CHUTE_ID_RE.match(chute_id or "")) and chute_id not in _PATH_TRAVERSAL


def require_valid_chute_id(chute_id: str) -> str:
    if not is_valid_chute_id(chute_id):
        raise RegistryError(f"not a usable chute id: {chute_id!r}")
    return chute_id


def is_valid_slug(slug: str) -> bool:
    """A DNS label **inside this subnet's namespace**.

    Both conditions, not either: the DNS rule keeps the slug usable as a
    hostname, and the prefix keeps it ours.
    """
    candidate = slug or ""
    return bool(_SLUG_RE.match(candidate)) and bool(_PREFIXED_SLUG_RE.match(candidate))


@dataclass(frozen=True)
class ChuteInfo:
    """What the platform reports about one chute."""

    chute_id: str
    slug: str
    hot: bool
    name: str = ""
    #: The deployed source when the info response carries it inline. Empty
    #: otherwise, in which case it is fetched separately.
    code: str = ""
    #: The image the deployment actually runs, as the platform reports it.
    #:
    #: The only handle there is. The API exposes an image's *identity* and never
    #: its contents — no dockerfile, no digest, no SBOM — and the id is
    #: ``uuid5(NAMESPACE_OID, "username/name:tag")``, a hash of three strings the
    #: miner chooses. So "is this the image the subnet published?" is the only
    #: question about it that can be answered, and the owner is checked beside it
    #: because an id alone says nothing about who built the thing behind it.
    image_id: str = ""
    image_username: str = ""


def _read_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _read_image(payload: Mapping[str, Any]) -> tuple[str, str]:
    """The deployed image's id and owner, or empty strings.

    Empty rather than raising: an absent image block is a platform schema
    change, and the eligibility check treats "unknown image" as "not the
    subnet's image", which is the safe direction.
    """
    image = payload.get("image")
    if not isinstance(image, dict):
        return "", ""
    owner = image.get("user")
    username = owner.get("username") if isinstance(owner, dict) else None
    return _read_str(image, "image_id"), username if isinstance(username, str) else ""


def _read_hot(payload: Mapping[str, Any], what: str) -> bool:
    """Whether the chute has live capacity.

    Accepts a flag or a count because the platform has reported both. Refusing
    to guess when neither is present matters: defaulting to ``True`` would score
    a dead deployment, and defaulting to ``False`` would zero a live miner over
    a schema change.
    """
    hot = payload.get("hot")
    if isinstance(hot, bool):
        return hot
    for key in ("hot_count", "instance_count"):
        count = payload.get(key)
        if isinstance(count, int) and not isinstance(count, bool):
            return count > 0
    instances = payload.get("instances")
    if isinstance(instances, list):
        # A *live* instance, not merely a listed one. Each entry carries its own
        # `active` and `verified` flags, and an instance that is neither is a
        # record of a deployment that is not serving. Counting the list length
        # made a chute full of dead instances read as hot, which would send a
        # batch of a hundred items at something that cannot answer and score the
        # miner incorrect for it. Absent flags are treated as live, because this
        # path only runs when the platform has stopped reporting `hot` at all
        # and inventing a stricter rule than the payload supports would zero a
        # working miner over a schema change.
        return any(_instance_is_live(entry) for entry in instances)
    raise RegistryError(f"{what} did not report whether it is hot")


def _instance_is_live(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return all(entry.get(flag) is not False for flag in ("active", "verified"))


def _extract_source(raw: bytes, what: str) -> str:
    """Decode a code response, unwrapping a JSON envelope if there is one."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegistryError(f"{what} is not UTF-8 text: {exc}") from exc

    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            envelope: Any = json.loads(stripped)
        except ValueError:
            envelope = None
        if isinstance(envelope, dict):
            for key in _SOURCE_KEYS:
                value = envelope.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            # A JSON object we cannot read source out of is an upstream schema
            # problem. Returning the envelope text would hash it as if it were
            # the miner's wrapper and blame the miner for it.
            raise RegistryError(f"{what} returned a JSON object carrying no source")

    if not text.strip():
        raise RegistryError(f"{what} is empty")
    return text


class ChutesClient(RegistryHttpClient):
    """Reads the Chutes API and builds inference endpoints from slugs."""

    def __init__(
        self,
        *,
        api_base: str = DEFAULT_CHUTES_API_BASE,
        endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if "{slug}" not in endpoint_template:
            raise RegistryError("endpoint_template must contain a {slug} placeholder")
        self._endpoint_template = endpoint_template
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        super().__init__(
            base_url=api_base,
            headers=headers,
            timeout_seconds=timeout_seconds,
            transport=transport,
            retries=retries,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )

    # -- fetching ---------------------------------------------------------

    def fetch_chute(self, chute_id: str) -> ChuteInfo:
        """Slug, hot status, and inline source if the platform supplies one."""
        require_valid_chute_id(chute_id)
        what = f"chute {chute_id}"
        payload = self.get_json(f"/chutes/{chute_id}", what=what)

        reported_id = _read_str(payload, "chute_id")
        if reported_id and reported_id != chute_id:
            raise RegistryError(
                f"{what} resolved to a different chute ({reported_id}); "
                "the on-chain commitment does not address this deployment"
            )

        slug = _read_str(payload, "slug")
        if not slug:
            raise RegistryError(f"{what} reports no slug, so it has no reachable endpoint")

        return ChuteInfo(
            chute_id=chute_id,
            slug=slug,
            hot=_read_hot(payload, what),
            name=_read_str(payload, "name"),
            code=_read_str(payload, "code"),
            image_id=_read_image(payload)[0],
            image_username=_read_image(payload)[1],
        )

    def fetch_deployed_source(self, chute_id: str) -> str:
        """The source the platform holds for this chute."""
        require_valid_chute_id(chute_id)
        what = f"deployed source of chute {chute_id}"
        raw = self.get_bytes(f"/chutes/code/{chute_id}", what=what, max_bytes=MAX_SOURCE_BYTES)
        return _extract_source(raw, what)

    def deployed_source(self, info: ChuteInfo) -> str:
        """The deployed source, reusing the info response when it carried one."""
        if info.code.strip():
            return info.code
        return self.fetch_deployed_source(info.chute_id)

    # -- addressing -------------------------------------------------------

    def endpoint_url(self, slug: str) -> str:
        """Base URL of a deployed chute, refusing a slug that is not a DNS label."""
        if not is_valid_slug(slug):
            raise RegistryError(
                f"not a usable chute slug: {slug!r}; a slug becomes a hostname, so it "
                f"must be a single DNS label and must start with {CHUTE_SLUG_PREFIX!r}- "
                "so the deployment is provably inside this subnet's namespace"
            )
        return self._endpoint_template.format(slug=slug)

    def moderate_url(self, slug: str) -> str:
        return self.endpoint_url(slug) + MODERATE_PATH

    def health_url(self, slug: str) -> str:
        return self.endpoint_url(slug) + HEALTH_PATH


def require_hot(info: ChuteInfo) -> ChuteInfo:
    if not info.hot:
        raise ChuteNotRunningError(
            f"chute {info.chute_id} ({info.slug}) is not hot, so it cannot serve /moderate"
        )
    return info


__all__ = [
    "DEFAULT_CHUTES_API_BASE",
    "DEFAULT_ENDPOINT_TEMPLATE",
    "HEALTH_PATH",
    "MAX_SOURCE_BYTES",
    "MODERATE_PATH",
    "ChuteInfo",
    "ChutesClient",
    "is_valid_chute_id",
    "CHUTE_SLUG_PREFIX",
    "is_valid_slug",
    "require_hot",
    "require_valid_chute_id",
]
