"""Copy mode is chosen under [validator], and copies the hotkey named there."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from prometheon.config import ValidatorMode, ValidatorRuntimeConfig

pytestmark = pytest.mark.unit

OWNER = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
PLACEHOLDER = "<owner validator hotkey, published with the launch>"


def test_score_mode_is_the_default_and_ignores_copy_from() -> None:
    config = ValidatorRuntimeConfig(copy_from=PLACEHOLDER)

    assert config.mode is ValidatorMode.SCORE


@pytest.mark.parametrize("copy_from", [PLACEHOLDER, "", "   ", "not-a-hotkey"])
def test_copy_mode_refuses_anything_but_a_hotkey(copy_from: str) -> None:
    """Refused at load, not at the first copy, which runs on a timer at night."""
    with pytest.raises(ValidationError, match="copy_from"):
        ValidatorRuntimeConfig(mode=ValidatorMode.COPY, copy_from=copy_from)


def test_copy_mode_with_the_owner_validators_hotkey_loads() -> None:
    config = ValidatorRuntimeConfig(mode=ValidatorMode.COPY, copy_from=OWNER)

    assert (config.mode, config.copy_from) == (ValidatorMode.COPY, OWNER)
