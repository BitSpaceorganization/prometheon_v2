"""The metagraph snapshot the validator reasons about.

Everything downstream (self-exclusion, scoring, the weight vector) is keyed by
**hotkey**, because a hotkey is identity and a UID is only its current address.
UIDs are recycled the moment a neuron deregisters, so a UID captured at the
start of a cycle can belong to somebody else fourteen hours later when weights
go out. This module is the single place hotkey → UID resolution happens, and
the runtime is expected to take a fresh snapshot immediately before submission
rather than reuse the one the cycle started with.

The view is frozen. The evaluation pipeline reads it and never edits it, so a
snapshot passed to scoring is the same snapshot that produced the weight
vector.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from prometheon.errors import SubtensorError

#: Rao per TAO. The SDK reports stake as TAO floats; this subnet stores rao
#: integers so no float can reach a payload that later gets canonicalised.
_RAO_PER_TAO: Final[int] = 1_000_000_000

#: Attribute names the SDK has used for the stake column, newest first.
_STAKE_ATTRIBUTES: Final[tuple[str, ...]] = ("total_stake", "stake", "S")


class MetagraphView(BaseModel):
    """A typed, immutable snapshot of the on-chain neuron table.

    Stored as parallel tuples rather than a list of objects because that is the
    shape the SDK hands back and the shape ``set_weights`` wants back; a row
    representation would mean two conversions for no gain.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    netuid: int = Field(ge=0)
    block: int = Field(default=0, ge=0)

    #: The subnet owner: the burn target when a reward pool has no winner.
    #: Carried on the snapshot because newer SDKs return it with the metagraph,
    #: which saves a second round trip for a value every cycle needs.
    owner_hotkey: str | None = None

    uids: tuple[int, ...] = ()
    hotkeys: tuple[str, ...] = ()
    #: Stake in rao (integer), never TAO floats.
    stake_rao: tuple[int, ...] = ()
    validator_permit: tuple[bool, ...] = ()

    _uid_by_hotkey: dict[str, int] = PrivateAttr(default_factory=dict)
    _row_by_uid: dict[int, int] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _build_index(self) -> MetagraphView:
        """Reject an inconsistent table, then index it once.

        A ragged snapshot cannot be papered over. If the stake column is one
        entry short, every hotkey after the gap is attributed the wrong stake,
        and nothing downstream can detect it.
        """
        count = len(self.hotkeys)
        for name, column in (
            ("uids", self.uids),
            ("stake_rao", self.stake_rao),
            ("validator_permit", self.validator_permit),
        ):
            if len(column) != count:
                raise ValueError(f"metagraph has {count} hotkeys but {len(column)} {name} entries")

        if len(set(self.uids)) != count:
            raise ValueError("metagraph uids are not unique")
        if len(set(self.hotkeys)) != count:
            raise ValueError("metagraph hotkeys are not unique")
        if any(stake < 0 for stake in self.stake_rao):
            raise ValueError("metagraph carries a negative stake")

        self._uid_by_hotkey.update(zip(self.hotkeys, self.uids, strict=True))
        self._row_by_uid.update({uid: row for row, uid in enumerate(self.uids)})
        return self

    def __len__(self) -> int:
        return len(self.hotkeys)

    def is_registered(self, hotkey: str) -> bool:
        return hotkey in self._uid_by_hotkey

    def uid_for(self, hotkey: str) -> int | None:
        """The current UID for ``hotkey``, or ``None`` if it is not registered."""
        return self._uid_by_hotkey.get(hotkey)

    def require_uid(self, hotkey: str) -> int:
        """Resolve ``hotkey`` or raise.

        Used for targets that must exist, above all the subnet owner hotkey a
        burn is routed to. Returning ``None`` there and letting the caller
        forget to check would silently drop the burn share.
        """
        uid = self._uid_by_hotkey.get(hotkey)
        if uid is None:
            raise SubtensorError(
                f"hotkey {hotkey!r} is not registered on netuid={self.netuid} at block {self.block}"
            )
        return uid

    def hotkey_for(self, uid: int) -> str | None:
        row = self._row_by_uid.get(uid)
        return None if row is None else self.hotkeys[row]

    def stake_rao_for(self, hotkey: str) -> int | None:
        row = self._row_for_hotkey(hotkey)
        return None if row is None else self.stake_rao[row]

    def has_validator_permit(self, hotkey: str) -> bool:
        row = self._row_for_hotkey(hotkey)
        return False if row is None else self.validator_permit[row]

    def validator_uids(self) -> tuple[int, ...]:
        return tuple(
            uid for uid, permit in zip(self.uids, self.validator_permit, strict=True) if permit
        )

    def partition_by_registration(self, hotkeys: list[str]) -> tuple[list[str], list[str]]:
        """Split ``hotkeys`` into ``(registered, unregistered)``, order preserved.

        A miner can deregister between the snapshot the cycle scored and the
        one weights are submitted against. The runtime uses this to drop those
        hotkeys and log how many vanished, instead of aborting a whole cycle's
        submission over somebody else's exit.
        """
        registered: list[str] = []
        unregistered: list[str] = []
        for hotkey in hotkeys:
            (registered if self.is_registered(hotkey) else unregistered).append(hotkey)
        return registered, unregistered

    def _row_for_hotkey(self, hotkey: str) -> int | None:
        uid = self._uid_by_hotkey.get(hotkey)
        return None if uid is None else self._row_by_uid[uid]

    @classmethod
    def from_sdk_metagraph(cls, raw: Any, *, netuid: int) -> MetagraphView:
        """Build a view from whatever the installed SDK's metagraph looks like.

        Duck-typed. The column names and element types have moved across SDK
        releases (stake as ``Balance`` objects, as numpy floats in TAO, under
        three different attribute names), and a validator should not need a new
        release to survive a rename it can detect at runtime.

        Two column layouts are read, because the SDK changed shape between
        major versions and a validator may be running either:

        - **A row list**: ``raw.neurons`` of objects carrying ``uid``,
          ``hotkey``, ``validator_permit`` and a ``Balance`` stake.
        - **Parallel arrays**: ``raw.hotkeys`` / ``raw.uids`` / ``raw.stake``,
          the older layout, with stake as numpy floats in TAO.

        What it will *not* do is guess past a length mismatch. See
        :meth:`_build_index`.
        """
        try:
            # `neurons` must be tested for *emptiness*, not for None. A
            # `Metagraph` carries `neurons = []` until it is synced with
            # `lite=False`, and `[] is not None`. Keying on None sent a
            # fully-populated parallel-array metagraph down the row path and
            # produced a zero-neuron view from a table with neurons in it.
            # That does not raise anywhere: it reads downstream as "every miner
            # deregistered". Caught by pushing a real SDK object through this
            # function; a hand-built fake had agreed with the bug.
            neurons = getattr(raw, "neurons", None) or None
            if neurons is not None:
                columns = _columns_from_rows(neurons)
            else:
                columns = _columns_from_arrays(raw)
            block = int(getattr(raw, "block", 0) or 0)
            owner = getattr(raw, "owner_hotkey", None)
        except (AttributeError, TypeError, ValueError) as exc:
            raise SubtensorError(
                f"metagraph for netuid={netuid} has an unexpected shape: {exc}"
            ) from exc

        try:
            return cls(
                netuid=netuid,
                block=block,
                owner_hotkey=str(owner) if owner else None,
                **columns,
            )
        except ValidationError as exc:
            raise SubtensorError(f"metagraph for netuid={netuid} is inconsistent: {exc}") from exc


def _columns_from_rows(neurons: Any) -> dict[str, tuple[Any, ...]]:
    """Transpose a list of neuron rows into the view's columns."""
    rows = list(neurons)
    return {
        "uids": tuple(int(row.uid) for row in rows),
        "hotkeys": tuple(str(row.hotkey) for row in rows),
        "stake_rao": tuple(_to_rao(_row_stake(row)) for row in rows),
        "validator_permit": tuple(bool(getattr(row, "validator_permit", False)) for row in rows),
    }


def _row_stake(row: Any) -> Any:
    for name in _STAKE_ATTRIBUTES:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return 0


def _columns_from_arrays(raw: Any) -> dict[str, tuple[Any, ...]]:
    hotkeys = [str(hotkey) for hotkey in getattr(raw, "hotkeys", [])]
    count = len(hotkeys)
    return {
        "uids": tuple(_coerce_uids(getattr(raw, "uids", None), count=count)),
        "hotkeys": tuple(hotkeys),
        "stake_rao": tuple(_coerce_stake(raw, count=count)),
        "validator_permit": tuple(
            _coerce_permits(getattr(raw, "validator_permit", None), count=count)
        ),
    }


def _coerce_uids(value: Any, *, count: int) -> list[int]:
    if value is None:
        # Some lite metagraphs omit the column; position is the UID there.
        return list(range(count))
    uids = [int(uid) for uid in value]
    if len(uids) != count:
        raise ValueError(f"metagraph has {count} hotkeys but {len(uids)} uids")
    return uids


def _coerce_stake(raw: Any, *, count: int) -> list[int]:
    """The first stake column that is actually populated, in rao.

    Candidates are tried in order and an *empty* one is treated as absent
    rather than as an error: a 10.x ``Metagraph`` exposes ``total_stake`` as an
    empty array alongside a populated ``stake``, so rejecting on the first
    length mismatch would fail every real metagraph. A column that is populated
    but the wrong length is genuine SDK drift, and it raises, because silently
    accepting it would attribute the wrong stake to every hotkey after the gap.
    """
    for name in _STAKE_ATTRIBUTES:
        column = getattr(raw, name, None)
        if column is None:
            continue
        stake = [_to_rao(entry) for entry in column]
        if not stake:
            continue
        if len(stake) != count:
            raise ValueError(f"metagraph has {count} hotkeys but {len(stake)} {name} entries")
        return stake
    # Stake decides nothing in this subnet's reward design, so an SDK that
    # stops publishing it costs a diagnostic field rather than a whole cycle.
    return [0] * count


def _to_rao(value: Any) -> int:
    rao = getattr(value, "rao", None)
    if rao is not None:
        return int(rao)
    return round(float(value) * _RAO_PER_TAO)


def _coerce_permits(value: Any, *, count: int) -> list[bool]:
    if value is None:
        return [False] * count
    permits = [bool(permit) for permit in value]
    if len(permits) != count:
        raise ValueError(f"metagraph has {count} hotkeys but {len(permits)} permit entries")
    return permits


__all__ = ["MetagraphView"]
