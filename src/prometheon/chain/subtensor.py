"""The bittensor SDK adapter, and the only module that imports it.

Every SDK call the validator makes goes through here. The import is lazy, so
the rest of the package (and its unit tests) never needs the SDK installed, and
the shape of what comes back is normalised once so one SDK release changing a
return type is one file to fix.

This adapter drives the 10.x named-accessor API, and `pyproject` pins
``bittensor>=10.5,<11`` to match. The 11.x client is a different library
wearing the same name: its ``Subtensor`` exposes a generic
``read(name, **params)`` / ``execute(intent, wallet)`` pair and *none* of
``set_weights``, ``metagraph``, ``get_commitment``, ``set_commitment`` or
``get_subnet_owner_hotkey``. More decisively, 11.x ships no intent that writes
a metadata commitment at all, so a miner on 11.x could not make a submission,
which is the one thing this subnet asks a miner to do.

An earlier revision of this file tried to support both generations behind
runtime feature detection. It shipped a bug, and that bug is why the dual path
is gone: on the 11.x branch the metagraph was fetched with
``read("metagraph")``, which returns a plain mapping rather than the typed
object the transposer expected. Every ``getattr`` missed and the transposer
produced a **zero-neuron view**. Nothing raised. Downstream it read as a
snapshot in which every miner was unregistered, so a cycle would have dropped
the entire field as ordinary churn and burned a day of emission while the logs
reported a normal run. Two paths, one of which nothing exercises, is how that
reaches production. Now there is one path, asserted at startup.

The adapter is paranoid about *absent* accessors and fails loud rather than
falling back. That is a second scar: an earlier version caught
``AttributeError`` around the hyperparameter read and defaulted
``commit_reveal_enabled`` to ``False``. When the subnet owner turned
commit-reveal on, the gate never noticed and submissions went somewhere nobody
was watching. A missing attribute means the SDK has drifted, and that has to
surface at startup, not as a wrong answer fourteen hours later.
"""

from __future__ import annotations

import inspect
from typing import Any, Final

from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import (
    ChainCapabilities,
    ChainHyperparameters,
    ChainWeightVector,
)
from prometheon.config import ChainNetwork
from prometheon.errors import SetWeightsFailedError, SubtensorError

#: Distinguishes "field absent from the SDK object", which is drift and fails
#: loud, from "field present but None", which is normal and means unset on
#: chain.
_MISSING: Final[object] = object()

#: How long a substrate receipt may run in an error message before it stops
#: being greppable.
_MAX_RECEIPT_CHARS: Final[int] = 300

#: The accessors this runtime drives. Checked together at connect time so an
#: incompatible SDK is one clear error at startup rather than five different
#: ``AttributeError``s spread across a fourteen-hour cycle.
_REQUIRED_ACCESSORS: Final[tuple[str, ...]] = (
    "metagraph",
    "get_subnet_hyperparameters",
    "get_subnet_owner_hotkey",
    "get_commitment",
    # Supplies the block a commitment was written in, which is what settles a
    # duplicate-model claim. Absent, duplicate resolution silently degrades to
    # uid order — the attack `registry.validation` exists to prevent.
    "get_commitment_metadata",
    "set_weights",
)

_PIN_ADVICE: Final[str] = (
    "this runtime drives the bittensor 10.x accessor API; pin 'bittensor>=10.5,<11'"
)


def _bittensor() -> Any:
    """Import the SDK, or raise a typed error naming the remedy.

    Indirected through a function so it happens at call time rather than import
    time, and so tests can substitute a fake without installing the SDK.
    """
    try:
        import bittensor
    except ImportError as exc:
        raise SubtensorError(
            "the bittensor SDK is not installed; install the package's chain "
            f"dependencies before running against a network: {exc}"
        ) from exc
    return bittensor


def _sdk_version() -> str:
    """Best-effort SDK version for diagnostics. Never raises."""
    try:
        import bittensor
    except ImportError:
        return "not-installed"
    return str(getattr(bittensor, "__version__", "unknown"))


def assert_sdk_compatible(subtensor: Any) -> None:
    """Fail loud if ``subtensor`` is not an SDK generation this runtime drives.

    Called once by :func:`connect`. The check is on the *methods*, not on
    ``__version__``: a version string tells you what was packaged, an attribute
    tells you what can actually be called, and a vendored or repackaged client
    can carry any version it likes.

    The message names every missing accessor at once. An 11.x client is missing
    all five, and an operator who fixes them one traceback at a time learns
    that fact five deploys later.
    """
    missing = [name for name in _REQUIRED_ACCESSORS if not callable(getattr(subtensor, name, None))]
    if not missing:
        return

    detail = ""
    if callable(getattr(subtensor, "read", None)) and callable(getattr(subtensor, "execute", None)):
        detail = (
            " This client exposes read()/execute(), which is the 11.x "
            "read-and-intent API. It is not a newer version of the same "
            "interface — it shares no accessor with 10.x, and it ships no "
            "intent that writes a metadata commitment, so a miner could not "
            "submit a model on it."
        )

    raise SubtensorError(
        f"the installed bittensor SDK ({_sdk_version()}) is missing "
        f"{', '.join(missing)}; {_PIN_ADVICE}.{detail}"
    )


def connect(network: ChainNetwork | str) -> Any:
    """Open a subtensor connection for ``network``, checked for compatibility.

    Returns ``Any``: the SDK's own types are dynamic, and pinning a type here
    would be a fiction that breaks on the next release.
    """
    name = network.value if isinstance(network, ChainNetwork) else str(network)
    if not name.strip():
        raise SubtensorError("network must be a non-empty identifier or chain endpoint")

    try:
        subtensor = _bittensor().Subtensor(network=name)
    except SubtensorError:
        raise
    except Exception as exc:
        raise SubtensorError(f"could not connect to subtensor on network={name!r}: {exc}") from exc

    assert_sdk_compatible(subtensor)
    return subtensor


def detect_capabilities(subtensor: Any | None = None) -> ChainCapabilities:
    """Detect whether this SDK lets us name the mechanism we are weighting.

    ``mechid`` selects which mechanism of a subnet the weights apply to. An SDK
    that does not take the argument submits to whatever the chain defaults to,
    which on a multi-mechanism subnet is an error nothing reports. Detection
    runs once at startup and feeds
    :func:`~prometheon.chain.weights.assert_submission_allowed`.
    """
    if subtensor is None:
        subtensor_class = getattr(_bittensor(), "Subtensor", None)
        if subtensor_class is None:
            raise SubtensorError(
                f"the installed bittensor SDK ({_sdk_version()}) exposes no "
                f"Subtensor class; {_PIN_ADVICE}"
            )
        target: Any = getattr(subtensor_class, "set_weights", None)
    else:
        target = getattr(subtensor, "set_weights", None)

    if target is None:
        raise SubtensorError(
            f"the installed bittensor SDK ({_sdk_version()}) exposes no "
            f"Subtensor.set_weights; {_PIN_ADVICE}"
        )

    return ChainCapabilities(sdk_version=_sdk_version(), supports_mechid=_accepts_mechid(target))


def sync_metagraph_view(subtensor: Any, *, netuid: int) -> MetagraphView:
    """Pull a fresh metagraph and normalise it into a :class:`MetagraphView`.

    ``lite=False`` is passed so the snapshot carries the stake and permit
    columns; the lite table omits them, and a permit column of all-``False``
    would silently mean "no validators on this subnet".
    """
    try:
        raw = subtensor.metagraph(netuid=netuid, lite=False)
    except TypeError:
        # An SDK that does not take `lite` still returns the full table.
        try:
            raw = subtensor.metagraph(netuid=netuid)
        except Exception as exc:
            raise SubtensorError(
                f"could not sync the metagraph for netuid={netuid}: {exc}"
            ) from exc
    except Exception as exc:
        raise SubtensorError(f"could not sync the metagraph for netuid={netuid}: {exc}") from exc

    if raw is None:
        raise SubtensorError(
            f"subtensor returned no metagraph for netuid={netuid}; the subnet may "
            "not exist on this network"
        )

    view = MetagraphView.from_sdk_metagraph(raw, netuid=netuid)

    # An empty table is never a legitimate answer for a live subnet: the owner
    # alone is always registered. Letting it through means every miner reads as
    # unregistered, which downstream is indistinguishable from the entire field
    # deregistering at once. That gets logged as churn and burns a day of
    # emission. Fail the cycle instead.
    if len(view) == 0:
        raise SubtensorError(
            f"the metagraph for netuid={netuid} came back with no neurons. A live "
            "subnet always has at least its owner registered, so this is a "
            "failed or misdirected read, not an empty subnet — refusing to "
            "treat every miner as deregistered"
        )
    return view


def read_hyperparameters(subtensor: Any, *, netuid: int) -> ChainHyperparameters:
    """Read the two hyperparameters the submission gates depend on.

    Both come from a single call. An earlier version used two convenience
    accessors that do not exist on ``Subtensor`` and hid the mistake behind
    ``except AttributeError``; see the module docstring. Called immediately
    before submission so the gates run on fresh values, not on whatever was
    true when the cycle started.
    """
    try:
        hyperparameters = subtensor.get_subnet_hyperparameters(netuid=netuid)
    except Exception as exc:
        raise SubtensorError(
            f"could not read subnet hyperparameters for netuid={netuid}: {exc}"
        ) from exc

    if hyperparameters is None:
        raise SubtensorError(
            f"subtensor returned no hyperparameters for netuid={netuid}; the subnet "
            "may not exist on this network"
        )

    commit_reveal = _require_field(
        hyperparameters, "commit_reveal_weights_enabled", owner="subnet hyperparameters"
    )
    weights_version = _require_field(
        hyperparameters, "weights_version", owner="subnet hyperparameters"
    )

    return ChainHyperparameters(
        # None means the subnet never set a version; the documented default is
        # 0, and the gate compares against a configured int that is never None.
        weights_version=0 if weights_version is None else int(weights_version),
        commit_reveal_enabled=bool(commit_reveal),
    )


def read_subnet_owner_hotkey(subtensor: Any, *, netuid: int) -> str:
    """Read the subnet owner hotkey: the burn target when a pool has no winner.

    Read from chain rather than configured, so every validator burns to the
    same place without anyone having to keep a config entry correct. A
    misconfigured burn target misroutes real emission.
    """
    try:
        owner = subtensor.get_subnet_owner_hotkey(netuid=netuid)
    except Exception as exc:
        raise SubtensorError(
            f"could not read the subnet owner hotkey for netuid={netuid}: {exc}"
        ) from exc

    if not isinstance(owner, str) or not owner:
        raise SubtensorError(
            f"the subnet owner hotkey for netuid={netuid} came back as {owner!r}; "
            "the subnet may not exist on this network"
        )
    return owner


def blocks_until_weights_allowed(subtensor: Any, *, netuid: int, hotkey: str) -> int:
    """How many blocks remain before this hotkey may set weights again.

    ``0`` when it may set them now, including when the chain cannot answer.
    Being wrong in that direction costs one rejected extrinsic; being wrong the
    other way would silently skip a submission that was due.

    Exists so a re-post between cycles can skip quietly instead of failing. The
    rate limit is short (100 blocks on netuid 108) and a re-post runs on a
    timer, so a cycle and a re-post landing minutes apart is routine, not an
    error worth waking anybody for.
    """
    try:
        limit = int(subtensor.weights_rate_limit(netuid=netuid))
    except Exception:  # pragma: no cover - older SDKs, or an RPC hiccup
        return 0
    if limit <= 0:
        return 0

    view = sync_metagraph_view(subtensor, netuid=netuid)
    try:
        uid = view.hotkeys.index(hotkey)
    except ValueError:
        # Not registered. Let the submission itself produce that error, which
        # says so far more clearly than a silent skip would.
        return 0

    try:
        elapsed = int(subtensor.blocks_since_last_update(netuid, uid))
    except Exception:  # pragma: no cover - as above
        return 0
    return max(0, limit - elapsed)


def submit_set_weights(
    subtensor: Any,
    *,
    wallet: Any,
    netuid: int,
    vector: ChainWeightVector,
    version_key: int,
    mechid: int | None,
    wait_for_inclusion: bool = True,
) -> str | None:
    """Submit the u16 vector, returning a best-effort receipt.

    ``mechid`` is passed only when it is not ``None``; the caller decides that
    from :func:`detect_capabilities`, so this function never re-inspects the
    SDK and stays trivially fakeable.

    Returns the extrinsic hash or id when the SDK surfaces one, otherwise the
    chain's status message, otherwise ``None``. Raises
    :class:`~prometheon.errors.SetWeightsFailedError` when the SDK *reports* a
    failure. The response is an ``ExtrinsicResponse`` dataclass with no
    ``__bool__``, so it is unconditionally truthy, and code that tested it for
    truth once swallowed real failures whole.
    """
    kwargs: dict[str, Any] = {
        "wallet": wallet,
        "netuid": netuid,
        "uids": list(vector.uids),
        "weights": list(vector.weights),
        "version_key": version_key,
        "wait_for_inclusion": wait_for_inclusion,
    }
    if mechid is not None:
        kwargs["mechid"] = mechid

    try:
        result = subtensor.set_weights(**kwargs)
    except SetWeightsFailedError:
        raise
    except Exception as exc:
        raise SetWeightsFailedError(f"set_weights failed on netuid={netuid}: {exc}") from exc

    return parse_extrinsic_response(result, operation="set_weights", error=SetWeightsFailedError)


def parse_extrinsic_response(
    result: Any,
    *,
    operation: str,
    error: type[Exception],
) -> str | None:
    """Normalise every shape a chain write has returned, raising on failure.

    Shared by weight submission and commitment publishing because both return
    ``ExtrinsicResponse`` on 10.x and both had the same failure-reads-as-success
    bug in their own copy of this logic.

    Current SDKs return the response dataclass; older ones returned
    ``(bool, str)``, a bare bool, or a hash string. Each is handled explicitly
    so a failure in any of them raises rather than passing silently. The tuple
    case matters most: a ``(False, "reason")`` tuple has no ``.success``
    attribute, so a check written as ``getattr(result, "success", True)``
    defaults it to ``True`` and reports a rejected write as a successful one.
    """
    if _looks_like_response(result):
        if not getattr(result, "success", False):
            raise error(f"{operation} reported failure: {_summarise_failure(result)}")
        return _best_receipt(result)

    if isinstance(result, tuple):
        if len(result) < 2:
            raise error(f"{operation} returned an uninterpretable {len(result)}-tuple")
        if not result[0]:
            raise error(f"{operation} reported failure: {result[1]!r}")
        return str(result[1]) if result[1] else None

    if isinstance(result, bool):
        if not result:
            raise error(f"{operation} returned False with no detail")
        return None

    if isinstance(result, str):
        return result

    if result is None:
        # Several older accessors returned nothing at all on success and raised
        # on failure. Nothing was signalled, so nothing is reported.
        return None

    raise error(
        f"{operation} returned {type(result).__name__}, which this runtime cannot "
        f"interpret as success or failure; refusing to assume it succeeded"
    )


def _looks_like_response(result: Any) -> bool:
    """True for an extrinsic-response object.

    Keyed on ``success`` alone. An earlier version required both ``success``
    and ``message``, so a response carrying ``success=False`` but no
    ``message`` fell through to the terminal branch and read as an unknown
    shape rather than as the failure it plainly was.
    """
    return hasattr(result, "success")


def _accepts_mechid(target: Any) -> bool:
    try:
        return "mechid" in inspect.signature(target).parameters
    except (TypeError, ValueError):
        # An unusual callable (a C extension, a proxy without a signature) tells
        # us nothing, so report the conservative answer and let the gate decide.
        return False


def _require_field(owner_object: Any, name: str, *, owner: str) -> Any:
    """Fetch ``name`` from an object or a mapping, failing loud when absent."""
    if isinstance(owner_object, dict):
        value = owner_object.get(name, _MISSING)
    else:
        value = getattr(owner_object, name, _MISSING)

    if value is _MISSING:
        raise SubtensorError(
            f"{owner} is missing {name!r}; the installed bittensor SDK "
            f"({_sdk_version()}) is not compatible with this runtime"
        )
    return value


def _best_receipt(result: Any) -> str | None:
    """The most specific chain identifier the result carries.

    Ordered by how well it identifies the extrinsic afterwards: a block-and-index
    identifier can be looked up on an explorer, a status message can only be read.

    The shape here is pinned to what the SDK actually returns, which is narrower
    than it looks. ``ExtrinsicResponse`` has **no** ``extrinsic_id``,
    ``extrinsic_hash`` or ``block_hash`` field, and the ``ExtrinsicReceipt`` it
    carries has no ``extrinsic_hash`` either. The identifier comes from
    ``get_extrinsic_identifier()``. An earlier version read three field names
    that exist on a different SDK generation, found none of them, and quietly
    returned the status message for every submission; operators got
    ``"Finalized."`` where they expected something they could look up.

    ``get_extrinsic_identifier`` is guarded because it can raise when the
    receipt has not resolved its block yet. A receipt is a diagnostic, so a
    failure to produce one must never fail a submission that the chain accepted.
    """
    receipt = getattr(result, "extrinsic_receipt", None)
    if receipt is not None:
        identify = getattr(receipt, "get_extrinsic_identifier", None)
        if callable(identify):
            try:
                identifier = identify()
            except Exception:
                identifier = None
            if identifier:
                return str(identifier)

        index = getattr(receipt, "extrinsic_idx", None)
        if index is not None:
            return f"extrinsic_idx={index}"

    # Kept for SDK shapes that do carry a flat identifier; harmless when absent.
    for name in ("extrinsic_id", "extrinsic_hash", "block_hash"):
        value = getattr(result, name, None)
        if value:
            return str(value)

    message = getattr(result, "message", None)
    return str(message) if message else None


def _summarise_failure(result: Any) -> str:
    """Join every diagnostic a failed response carries.

    ``.message`` is sometimes empty on failure, and the real rejection (rate
    limit, missing permit, fee too low) surfaces only on ``.error`` or the
    receipt. Printing just the message leaves operators staring at ``None``.
    """
    parts: list[str] = []
    for name in ("message", "error", "extrinsic_function", "extrinsic_fee", "extrinsic_id"):
        value = getattr(result, name, None)
        if value is not None and value != "":
            parts.append(f"{name}={value!r}")

    receipt = getattr(result, "extrinsic_receipt", None)
    if receipt is not None:
        text = str(receipt)
        if len(text) > _MAX_RECEIPT_CHARS:
            text = text[: _MAX_RECEIPT_CHARS - 3] + "..."
        parts.append(f"receipt={text}")

    return "; ".join(parts) if parts else "the SDK provided no detail"


__all__ = [
    "assert_sdk_compatible",
    "connect",
    "detect_capabilities",
    "parse_extrinsic_response",
    "read_hyperparameters",
    "read_subnet_owner_hotkey",
    "submit_set_weights",
    "sync_metagraph_view",
]
