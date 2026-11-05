"""Regression tests for the smaller defects in the v2 notes, one per fix."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.unit


class TestTheRateLimitWaitsOneBlockLonger:
    """B16: the chain allows a new set only after *more* than the limit has passed."""

    def _waiting(self, monkeypatch: pytest.MonkeyPatch, *, limit: int, elapsed: int) -> int:
        from prometheon.chain import subtensor as chain

        view = SimpleNamespace(hotkeys=["5Validator"])
        monkeypatch.setattr(chain, "sync_metagraph_view", lambda *_a, **_k: view)
        node = SimpleNamespace(
            weights_rate_limit=lambda netuid: limit,
            blocks_since_last_update=lambda netuid, uid: elapsed,
        )
        return chain.blocks_until_weights_allowed(node, netuid=1, hotkey="5Validator")

    def test_exactly_at_the_limit_one_more_block_is_needed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._waiting(monkeypatch, limit=100, elapsed=100) == 1

    def test_past_the_limit_nothing_is_needed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._waiting(monkeypatch, limit=100, elapsed=101) == 0

    def test_well_inside_the_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._waiting(monkeypatch, limit=100, elapsed=40) == 61


def test_the_registry_package_exports_only_what_exists() -> None:
    """B25: `__all__` named `require_hot`, so a star import raised."""
    import prometheon.registry as registry

    for name in registry.__all__:
        assert hasattr(registry, name), name


class TestTheTemperatureGate:
    """B30: the provider-prefixed id that OpenRouter uses slipped past the gate."""

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "openai/gpt-5.6-sol"])
    def test_zero_is_refused_for_the_gpt5_line_with_or_without_a_prefix(self, model: str) -> None:
        from prometheon.config import LabellingConfig

        with pytest.raises(ValueError, match="temperature"):
            LabellingConfig(model=model, temperature=0)

    def test_zero_is_fine_for_another_model(self) -> None:
        from prometheon.config import LabellingConfig

        assert LabellingConfig(model="openai/gpt-4.1", temperature=0).temperature == 0


class _CharTokenizer:
    """One token per character, so lengths are easy to reason about."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class TestALongItemKeepsItsAnswerCue:
    """B12: v2 cut the tail of a long item's prompt, the cue included."""

    def test_a_short_item_is_encoded_as_one_string(self) -> None:
        from prometheon.evaluation.engine import ITEM_PROMPT, encode_item

        tokenizer = _CharTokenizer()
        assert encode_item(tokenizer, "hello") == tokenizer.encode(
            ITEM_PROMPT.format(content="hello")
        )

    def test_a_long_item_is_cut_inside_its_content_and_still_ends_with_the_cue(self) -> None:
        from prometheon.evaluation.engine import MAX_CONTENT_TOKENS, encode_item

        tokenizer = _CharTokenizer()
        ids = encode_item(tokenizer, "x" * (MAX_CONTENT_TOKENS * 3))
        text = "".join(chr(token) for token in ids)
        assert text.endswith("=== END CONTENT ===\n\nAnswer:")
        assert text.count("x") == MAX_CONTENT_TOKENS


class TestNoTracebackReachesAnOperator:
    """B24: anything this package failed to type still renders as one line."""

    def test_an_unexpected_error_is_one_line_and_exit_one(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli_main = importlib.import_module("prometheon.cli.main")

        def broken(_args: Any) -> int:
            raise ValueError("something nobody anticipated")

        parser = cli_main.build_parser()
        monkeypatch.setattr(cli_main, "build_parser", lambda: parser)
        monkeypatch.delenv("PROMETHEON_DEBUG", raising=False)
        monkeypatch.setattr(
            parser, "parse_args", lambda _argv=None: SimpleNamespace(handler=broken)
        )

        assert cli_main.main(["anything"]) == 1
        err = capsys.readouterr().err
        assert "error [internal] ValueError: something nobody anticipated" in err
        assert "Traceback" not in err

    def test_the_traceback_is_there_when_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cli_main = importlib.import_module("prometheon.cli.main")

        def broken(_args: Any) -> int:
            raise ValueError("boom")

        parser = cli_main.build_parser()
        monkeypatch.setattr(cli_main, "build_parser", lambda: parser)
        monkeypatch.setattr(
            parser, "parse_args", lambda _argv=None: SimpleNamespace(handler=broken)
        )
        monkeypatch.setenv("PROMETHEON_DEBUG", "1")
        with pytest.raises(ValueError, match="boom"):
            cli_main.main(["anything"])
