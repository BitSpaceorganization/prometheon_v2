"""What a validator remembers between processes: the day's vector and record.

One file, ``{state_directory}/last-weights.json``, written atomically at every
step a cycle completes, so a crash or a restart never costs more than the step
it interrupted:

1. **computed**: the vector and the unsigned record are saved *before* anything
   is sent. v2 saved nothing until the send succeeded, so a send refused inside
   the rate limit lost the whole day at the last step (B3).
2. **sent**: the units actually sent, and which hotkeys had deregistered and
   went to burn instead.
3. **signed**: the record, signed once and saved before it is posted. An
   SR25519 signature is randomised, so re-signing makes a different record that
   the hub refuses; the saved bytes are posted again instead (B4).
4. **published**: the hub accepted it.

``validator run`` for a day the file already holds resumes from the first step
not done, instead of labelling and evaluating the day again. ``validator
resubmit`` re-posts the vector, and posts a record still waiting to be
published.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from prometheon.config import Config
from prometheon.errors import ConfigError

STATE_FILE: Final[str] = "last-weights.json"


@dataclass
class CycleState:
    day: dt.date
    netuid: int
    burn_hotkey: str
    scoring_version: str
    #: The vector the cycle computed, burn included. What re-posts send.
    weights: dict[str, int]
    computed_at: int
    sent_at: int | None = None
    #: Units as actually sent: deregistered hotkeys' units moved to burn, and
    #: the vector shaped to the chain's weight limits.
    sent_weights: dict[str, int] | None = None
    #: The uid of each hotkey in ``sent_weights`` when it was sent. The record
    #: needs it for a uid that was added only to meet the chain's minimum.
    sent_uids: dict[str, int] | None = None
    #: Hotkeys that had deregistered by the time weights were sent.
    gone: list[str] = field(default_factory=list)
    receipt: str | None = None
    #: The results record as JSON: unsigned until ``signed`` is set.
    record: dict[str, Any] | None = None
    signed: bool = False
    published_at: int | None = None

    def is_complete(self, config: Config) -> bool:
        """Whether nothing is left to do for this day under ``config``."""
        sent = self.sent_at is not None or not config.validator.submit_weights
        published = (
            self.published_at is not None
            or not config.validator.submit_results
            or self.record is None
        )
        return sent and published


def state_path(config: Config) -> Path:
    return config.validator.state_directory / STATE_FILE


def load_state(config: Config) -> CycleState | None:
    """The saved state, ``None`` if there is none, or a typed error if unreadable."""
    path = state_path(config)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _from_json(raw)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # v2 let these escape as tracebacks, one of them after weights were
        # sent (B24). The file is written atomically now, so a corrupt one
        # means something outside this program touched it.
        raise ConfigError(
            f"{path} cannot be read ({type(exc).__name__}: {exc}). Move it aside and run "
            "`prometheon validator run` to compute the day again"
        ) from exc


def save_state(config: Config, state: CycleState) -> Path:
    """Write the state atomically: a crash leaves the old file or the new one."""
    path = state_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_to_json(state), indent=2, sort_keys=True) + "\n"
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".last-weights.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


def _to_json(state: CycleState) -> dict[str, Any]:
    data = asdict(state)
    data["day"] = state.day.isoformat()
    return data


def _from_json(raw: Any) -> CycleState:
    if not isinstance(raw, dict):
        raise TypeError("the state file is not a JSON object")
    weights = {str(key): int(value) for key, value in dict(raw["weights"]).items()}
    sent = raw.get("sent_weights")
    sent_uids = raw.get("sent_uids")
    # A v2 file carries `submitted_at` and no `sent_at`: its vector was sent.
    sent_at = raw.get("sent_at", raw.get("submitted_at"))
    return CycleState(
        day=dt.date.fromisoformat(str(raw["day"])),
        netuid=int(raw["netuid"]),
        burn_hotkey=str(raw["burn_hotkey"]),
        scoring_version=str(raw.get("scoring_version", "")),
        weights=weights,
        computed_at=int(raw.get("computed_at") or sent_at or 0),
        sent_at=int(sent_at) if sent_at is not None else None,
        sent_weights=(
            {str(key): int(value) for key, value in dict(sent).items()} if sent else None
        ),
        sent_uids=(
            {str(key): int(value) for key, value in dict(sent_uids).items()} if sent_uids else None
        ),
        gone=[str(hotkey) for hotkey in raw.get("gone", [])],
        receipt=raw.get("receipt"),
        record=raw.get("record"),
        signed=bool(raw.get("signed", False)),
        published_at=raw.get("published_at"),
    )


__all__ = ["STATE_FILE", "CycleState", "load_state", "save_state", "state_path"]
