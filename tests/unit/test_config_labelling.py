"""The labelling defaults have to be runnable as shipped."""

from __future__ import annotations

import pytest

from prometheon.config import LabellingConfig

pytestmark = pytest.mark.unit


class TestTheLabellingDefaultsCanActuallyRun:
    """A default that cannot run beside the neighbouring default is a trap.

    The blocker this exists for: `model` defaulted to a gpt-5 reasoning model
    and `temperature` to 0, and that line rejects an explicit 0 with HTTP 400.
    Nothing caught it until a cycle was hours in, past the snapshot fetch and
    into the labelling bill, and the provider's error named neither the config
    file nor the fix.
    """

    def test_the_shipped_defaults_are_mutually_compatible(self) -> None:
        config = LabellingConfig()
        assert not (config.model.startswith("gpt-5") and config.temperature == 0)

    def test_a_rejected_pair_fails_at_load_not_mid_cycle(self) -> None:
        with pytest.raises(ValueError, match="rejects an explicit temperature of 0"):
            LabellingConfig(temperature=0.0)

    def test_zero_stays_available_where_the_model_accepts_it(self) -> None:
        """0 is still the right value: it is what stops ground truth wandering."""
        assert LabellingConfig(model="gpt-4.1", temperature=0.0).temperature == 0.0

    def test_omitting_the_field_entirely_is_allowed(self) -> None:
        assert LabellingConfig(temperature=None).temperature is None
