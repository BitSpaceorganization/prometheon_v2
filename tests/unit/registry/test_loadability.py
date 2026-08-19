"""An architecture the image cannot load must fail before it costs anything."""

from __future__ import annotations

import json
from typing import Any

import pytest

from prometheon.errors import RegistryError
from prometheon.registry.loadability import (
    ArchitectureUnsupportedError,
    known_model_types,
    model_type_of,
    require_loadable,
)

pytestmark = pytest.mark.unit


class FakeClient:
    """Serves one `config.json`, or raises the way the real client does."""

    def __init__(self, payload: Any = None, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def fetch_file(self, repo: str, revision: str, filename: str, *, max_bytes: int = 0) -> bytes:
        assert filename == "config.json"
        if self.fail:
            raise RegistryError("not found")
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode()


class TestReadingTheDeclaredArchitecture:
    def test_a_declared_model_type_is_read(self) -> None:
        assert model_type_of(FakeClient({"model_type": "qwen2"}), "r", "a" * 40) == "qwen2"

    def test_an_unreadable_config_is_not_an_error_here(self) -> None:
        """The manifest check reports a broken repository, and reports it better."""
        assert model_type_of(FakeClient(fail=True), "r", "a" * 40) == ""

    def test_malformed_json_is_not_an_error_here(self) -> None:
        assert model_type_of(FakeClient(b"{not json"), "r", "a" * 40) == ""

    def test_a_config_without_the_field_reads_as_unknown(self) -> None:
        assert model_type_of(FakeClient({"architectures": ["X"]}), "r", "a" * 40) == ""


@pytest.fixture
def pinned_transformers(monkeypatch: Any) -> None:
    """What `transformers` 4.46 knows, which is what the image pins.

    Pinned rather than read from the installed package: `transformers` is an
    optional extra, and a test that silently turns into "unchecked" when it is
    absent would assert nothing at all.
    """
    monkeypatch.setattr(
        "prometheon.registry.loadability.known_model_types",
        lambda: frozenset({"llama", "mistral", "qwen2", "gemma2", "phi3"}),
        raising=True,
    )


@pytest.mark.usefixtures("pinned_transformers")
class TestRefusingWhatTheImageCannotLoad:
    """The blocker this exists for.

    `Qwen3-14B` was committed and deployed. Every instance died inside the
    container at `AutoModelForCausalLM.from_pretrained` with `KeyError:
    'qwen3'`, because the image pins `transformers<4.47` and Qwen3 landed in
    4.51. Nothing outside the container reported a reason: eleven instances
    were assigned and torn down over ninety minutes, which reads as missing GPU
    capacity rather than a model that cannot load.
    """

    def test_an_architecture_the_pin_predates_is_refused(self) -> None:
        with pytest.raises(ArchitectureUnsupportedError, match="qwen3"):
            require_loadable(FakeClient({"model_type": "qwen3"}), "org/model", "a" * 40)

    def test_the_message_says_what_to_do_about_it(self) -> None:
        with pytest.raises(ArchitectureUnsupportedError) as caught:
            require_loadable(FakeClient({"model_type": "qwen3"}), "org/model", "a" * 40)
        message = str(caught.value)
        assert "org/model" in message
        assert "Choose a checkpoint" in message

    def test_a_supported_architecture_passes(self) -> None:
        assert require_loadable(FakeClient({"model_type": "qwen2"}), "r", "a" * 40) == "qwen2"

    def test_an_undeclared_architecture_is_not_blocked(self) -> None:
        """Only refuse what is known to be unloadable; the manifest owns the rest."""
        assert require_loadable(FakeClient({}), "r", "a" * 40) == ""


class TestTheFrozenMappingIsTheImages:
    """The list is frozen on purpose, and that is the load-bearing decision.

    Reading the *installed* transformers would answer the wrong question. A
    miner running 4.51 locally would see `qwen3` accepted and still watch every
    instance die inside a container pinned to 4.46. The check has to describe
    the image, so it ships with the package instead of being discovered at
    runtime.
    """

    def test_it_ships_and_is_not_empty(self) -> None:
        known = known_model_types()
        assert len(known) > 100

    def test_it_matches_the_pin_that_broke_us(self) -> None:
        known = known_model_types()
        assert "qwen2" in known, "Qwen2.5 loads on the pinned image"
        assert "qwen3" not in known, "Qwen3 arrived in transformers 4.51; the image pins <4.47"

    def test_it_is_not_read_from_the_local_environment(self) -> None:
        """Whatever is installed here must not change the answer."""
        import sys
        from types import SimpleNamespace

        fake = SimpleNamespace(CONFIG_MAPPING_NAMES={"qwen3": "X"})
        saved = sys.modules.get("transformers.models.auto.configuration_auto")
        sys.modules["transformers.models.auto.configuration_auto"] = fake  # type: ignore[assignment]
        try:
            assert "qwen3" not in known_model_types()
        finally:
            if saved is None:
                sys.modules.pop("transformers.models.auto.configuration_auto", None)
            else:
                sys.modules["transformers.models.auto.configuration_auto"] = saved
