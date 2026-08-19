"""The architecture check that turns an invisible failure into a refusal.

The blocker behind this file: a checkpoint whose `model_type` the pinned
transformers does not know is accepted at commit, deploys successfully, gets an
instance, reports `verified: true`, and then dies during startup inside the
container. Eleven instances came and went over ninety minutes with no failure
reason on the API, which reads as missing GPU capacity rather than a model that
cannot load.
"""

from __future__ import annotations

import json

import pytest

from prometheon.errors import RegistryError
from prometheon.registry.loadability import (
    ArchitectureUnsupportedError,
    known_model_types,
    model_type_of,
    require_loadable,
)

pytestmark = pytest.mark.unit

REPO = "someone/guard"
REV = "a" * 40


class FakeClient:
    """Stands in for HuggingFaceClient.fetch_file."""

    def __init__(self, payload: bytes | Exception) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, str]] = []

    def fetch_file(self, repo: str, revision: str, filename: str, *, max_bytes: int = 0) -> bytes:
        self.calls.append((repo, revision, filename))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _config(model_type: str) -> bytes:
    return json.dumps({"model_type": model_type, "architectures": ["X"]}).encode()


class TestReadingTheDeclaredArchitecture:
    def test_it_reads_model_type_from_config_json(self) -> None:
        client = FakeClient(_config("qwen2"))
        assert model_type_of(client, REPO, REV) == "qwen2"
        assert client.calls == [(REPO, REV, "config.json")]

    def test_an_unreadable_config_is_not_this_check_s_problem(self) -> None:
        """The manifest check reports a missing config.json far better."""
        assert model_type_of(FakeClient(RegistryError("404")), REPO, REV) == ""

    def test_malformed_json_does_not_raise(self) -> None:
        assert model_type_of(FakeClient(b"{not json"), REPO, REV) == ""

    def test_a_config_without_model_type_reads_as_unknown(self) -> None:
        assert model_type_of(FakeClient(b'{"architectures": ["X"]}'), REPO, REV) == ""


class TestRefusingAnArchitectureTheImageCannotLoad:
    """These never skip.

    `transformers` is an optional extra, so a test gated on it not being
    installed would silently never run -- and this repository's own rule is
    that a boundary test which skips is not a boundary test. The known set is
    injected instead, which tests the refusal rather than the local venv.
    """

    #: What transformers 4.44-4.46 recognises, near enough. `qwen3` arrived in
    #: 4.51 and is deliberately absent.
    PINNED = frozenset({"qwen2", "llama", "mistral", "gemma2", "phi3"})

    def test_the_model_that_actually_broke_us_is_refused(self) -> None:
        """`qwen3` against a transformers that predates it.

        Not hypothetical: this was deployed, committed, and left crash-looping
        for hours before anyone could see why.
        """
        with pytest.raises(ArchitectureUnsupportedError, match="qwen3"):
            require_loadable(FakeClient(_config("qwen3")), REPO, REV, known=self.PINNED)

    def test_a_supported_architecture_passes(self) -> None:
        assert (
            require_loadable(FakeClient(_config("qwen2")), REPO, REV, known=self.PINNED) == "qwen2"
        )

    def test_an_undeclared_architecture_is_not_refused(self) -> None:
        """Silence is not evidence of a bad model, and this check does not guess."""
        assert require_loadable(FakeClient(b"{}"), REPO, REV, known=self.PINNED) == ""

    def test_without_transformers_it_reports_unchecked_rather_than_passing(self) -> None:
        """Claiming a checkpoint loads when nothing verified it is the bug itself."""
        assert require_loadable(FakeClient(_config("qwen3")), REPO, REV, known=None) is None

    def test_the_error_says_what_the_failure_would_look_like(self) -> None:
        """The message has to beat the symptom, which looks like missing capacity."""
        with pytest.raises(ArchitectureUnsupportedError) as caught:
            require_loadable(FakeClient(_config("qwen3")), REPO, REV, known=self.PINNED)

        message = str(caught.value)
        assert "die during startup" in message
        assert "no failure reason" in message


class TestTheRealTransformersIsConsulted:
    def test_the_installed_mapping_is_read_when_nothing_is_injected(self) -> None:
        """Without an override it must consult the real library, not a guess."""
        known = known_model_types()
        if known is None:
            # transformers absent: the contract is "unchecked", never "fine".
            assert require_loadable(FakeClient(_config("qwen3")), REPO, REV) is None
        else:
            assert "llama" in known
