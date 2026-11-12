"""A computed day survives the process that computed it (B3, B4, B5).

v2 saved the vector only after a successful send, so a send refused inside the
rate limit lost the whole day at the last step. Its record was signed and posted
once, so a lost response lost the record. And the record described the vector
as computed, not as sent. These tests drive ``validator run`` and ``validator
resubmit`` with a scripted chain and the fake hub, and check what is on disk and
on the wire after each step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair

from prometheon.chain.metagraph import MetagraphView
from prometheon.chain.weights import ChainHyperparameters
from prometheon.cli import validator as cli
from prometheon.cli.record import finish_record
from prometheon.cli.state import CycleState, load_state, save_state, state_path
from prometheon.config import (
    ChainConfig,
    ChainNetwork,
    Config,
    HubConfig,
    ValidatorRuntimeConfig,
    WalletConfig,
)
from prometheon.errors import ConfigError, ScoringError, SetWeightsFailedError
from prometheon.hub.auth import CallerRole, verify_evaluation
from prometheon.hub.client import HubClient
from prometheon.hub.fake import FakeHub
from prometheon.hub.models import EvaluationSubmission, MinerResult, ModelStatus
from prometheon.version import SCORING_VERSION

pytestmark = pytest.mark.unit

NETUID = 481
TODAY = dt.datetime.now(dt.timezone.utc).date()
DAY = TODAY - dt.timedelta(days=1)
VALIDATOR = Keypair.create_from_uri("//DurableValidator")
OWNER = Keypair.create_from_uri("//DurableOwner").ss58_address
MINER_A = Keypair.create_from_uri("//DurableA").ss58_address
MINER_B = Keypair.create_from_uri("//DurableB").ss58_address


def _row(hotkey: str, uid: int, weight: int) -> MinerResult:
    return MinerResult(
        hotkey=hotkey,
        uid=uid,
        weight=weight,
        dataset_submitted_count=10,
        dataset_valid_count=10,
        dataset_violating_confirmed=5,
        dataset_compliant_confirmed=5,
        dataset_score_micro=10_000_000,
        model_status=ModelStatus.EVALUATED,
        model_items_scored=100,
        model_correct_count=90,
        raw_accuracy_bp=9000,
        model_score_bp=9000,
        model_rank=1,
    )


def _draft(epoch_hash: str) -> EvaluationSubmission:
    return EvaluationSubmission(
        date=DAY,
        netuid=NETUID,
        validator_hotkey=VALIDATOR.ss58_address,
        scoring_version=SCORING_VERSION,
        policy_version="2026-08-07",
        epoch_content_hash=epoch_hash,
        corpus_content_hash="c" * 64,
        labelled_test_count=20,
        labelled_production_count=30,
        results=(_row(MINER_A, 1, 600_000), _row(MINER_B, 2, 390_000)),
        burned_weight=10_000,
        started_at=1,
        completed_at=2,
    )


WEIGHTS = {MINER_A: 600_000, MINER_B: 390_000, OWNER: 10_000}


class Chain:
    """Just enough chain: a metagraph, the gates, and a scripted set_weights."""

    def __init__(self, *, registered: list[str], refuse_sends: int = 0) -> None:
        self.registered = registered
        self.refuse_sends = refuse_sends
        self.sent: list[dict[str, Any]] = []
        self.wait_calls = 0

    def view(self) -> MetagraphView:
        return MetagraphView(
            netuid=NETUID,
            block=10,
            uids=tuple(range(len(self.registered))),
            hotkeys=tuple(self.registered),
            stake_rao=(1,) * len(self.registered),
            validator_permit=(True,) + (False,) * (len(self.registered) - 1),
        )

    def set_weights(self, **kwargs: Any) -> Any:
        if self.refuse_sends:
            self.refuse_sends -= 1
            raise SetWeightsFailedError("the rate limit is not clear")
        self.sent.append(kwargs)
        return "0xreceipt"


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    hub = FakeHub(netuid=NETUID)
    hub.register(VALIDATOR.ss58_address, CallerRole.VALIDATOR)
    raw = hub.publish_day(
        DAY,
        miners=[
            {
                "hotkey": VALIDATOR.ss58_address,
                "uid": 0,
                "coldkey": VALIDATOR.ss58_address,
                "verified_active_members": 0,
                "members": 0,
                "eligible": False,
            }
        ],
    )
    from prometheon.hub.epoch import epoch_content_hash

    chain = Chain(registered=[OWNER, MINER_A, MINER_B, VALIDATOR.ss58_address])
    config = Config(
        chain=ChainConfig(network=ChainNetwork.TEST, netuid=NETUID),
        wallet=WalletConfig(name="validator", hotkey="default"),
        hub=HubConfig(base_url="https://hub.invalid", epoch_keys=hub.epoch_keys),
        validator=ValidatorRuntimeConfig(state_directory=tmp_path / "state"),
    )
    computed: list[str] = []

    def fake_run_cycle(**_kwargs: Any) -> Any:
        computed.append("cycle")
        return SimpleNamespace(
            split=SimpleNamespace(weights=dict(WEIGHTS)),
            burn_hotkey=OWNER,
            notes=(),
        )

    def client(config: Any, *, keypair: Any = None) -> HubClient:
        return HubClient(
            base_url="https://hub.invalid",
            netuid=NETUID,
            epoch_keys=hub.epoch_keys,
            keypair=keypair,
            # The fake's handler is looked up per request, so a test can swap
            # it, and one retry is allowed, as in production.
            transport=httpx.MockTransport(lambda request: hub.handler(request)),
            max_retries=1,
            sleep=lambda _seconds: None,
        )

    monkeypatch.setattr(cli, "load", lambda _path: config)
    monkeypatch.setattr(cli, "open_wallet", lambda _config: SimpleNamespace(hotkey=VALIDATOR))
    monkeypatch.setattr(cli, "_policy_text", lambda _args: SimpleNamespace(version="2026-08-07"))
    monkeypatch.setattr(cli, "require_env", lambda *a, **k: "key")
    monkeypatch.setattr(cli, "run_cycle", fake_run_cycle)
    monkeypatch.setattr(cli, "_summary", lambda *_a: "summary")
    monkeypatch.setattr(cli, "build_results", lambda _result: ())
    monkeypatch.setattr(cli, "draft_record", lambda *_a, **_k: _draft(epoch_content_hash(raw)))
    monkeypatch.setattr(cli, "_hub_client", client)
    monkeypatch.setattr(cli, "HuggingFaceClient", lambda: _Context())
    monkeypatch.setattr(cli, "OpenAICompatibleClient", lambda _config: _Context())
    monkeypatch.setattr(cli.chain, "connect", lambda _network: chain)
    monkeypatch.setattr(cli.chain, "sync_metagraph_view", lambda *_a, **_k: chain.view())
    monkeypatch.setattr(
        cli.chain,
        "detect_capabilities",
        lambda *_a: SimpleNamespace(supports_mechid=True, sdk_version="test"),
    )
    monkeypatch.setattr(
        cli.chain,
        "read_hyperparameters",
        lambda *_a, **_k: ChainHyperparameters(weights_version=0, commit_reveal_enabled=False),
    )
    monkeypatch.setattr(
        "prometheon.cli.cycle.chain.read_hyperparameters",
        lambda *_a, **_k: ChainHyperparameters(weights_version=0, commit_reveal_enabled=False),
    )
    monkeypatch.setattr(
        "prometheon.cli.cycle.chain.detect_capabilities",
        lambda *_a: SimpleNamespace(supports_mechid=True, sdk_version="test"),
    )
    monkeypatch.setattr(
        "prometheon.cli.cycle.chain.sync_metagraph_view", lambda *_a, **_k: chain.view()
    )
    monkeypatch.setattr(
        "prometheon.cli.cycle.chain.submit_set_weights",
        lambda _subtensor, **kwargs: chain.set_weights(**kwargs),
    )

    def wait(*_a: Any, **_k: Any) -> None:
        chain.wait_calls += 1

    monkeypatch.setattr(cli, "_wait_for_weights_window", wait)
    return {"hub": hub, "chain": chain, "config": config, "computed": computed}


class _Context:
    def __enter__(self) -> _Context:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _run(**options: Any) -> int:
    return cli.cmd_run(
        argparse.Namespace(config="cfg.toml", date=DAY.isoformat(), policy=None, **options)
    )


def _resubmit() -> int:
    return cli.cmd_resubmit(argparse.Namespace(config="cfg.toml", date=None))


class TestTheVectorIsSavedBeforeItIsSent:
    def test_a_refused_send_keeps_the_day(self, world: dict[str, Any]) -> None:
        """B3: the computed vector is on disk although the send failed."""
        world["chain"].refuse_sends = 1
        with pytest.raises(SetWeightsFailedError):
            _run()

        state = load_state(world["config"])
        assert state is not None
        assert state.weights == WEIGHTS
        assert state.sent_at is None

    def test_the_next_run_resumes_instead_of_computing_again(self, world: dict[str, Any]) -> None:
        world["chain"].refuse_sends = 1
        with pytest.raises(SetWeightsFailedError):
            _run()
        assert _run() == 0

        assert world["computed"] == ["cycle"], "the day must not be labelled twice"
        assert len(world["chain"].sent) == 1
        state = load_state(world["config"])
        assert state is not None and state.sent_at is not None and state.published_at is not None

    def test_the_cycle_waits_for_the_rate_limit_before_sending(self, world: dict[str, Any]) -> None:
        _run()
        assert world["chain"].wait_calls == 1

    def test_a_finished_day_is_not_run_again(self, world: dict[str, Any]) -> None:
        _run()
        _run()
        assert world["computed"] == ["cycle"]
        assert len(world["chain"].sent) == 1

    def test_force_computes_the_day_again(self, world: dict[str, Any]) -> None:
        _run()
        _run(force=True)
        assert world["computed"] == ["cycle", "cycle"]

    def test_a_re_post_sends_a_vector_whose_cycle_never_sent_it(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        world["chain"].refuse_sends = 1
        with pytest.raises(SetWeightsFailedError):
            _run()
        monkeypatch.setattr(cli.chain, "blocks_until_weights_allowed", lambda *_a, **_k: 0)
        assert _resubmit() == 0
        state = load_state(world["config"])
        assert state is not None and state.sent_at is not None


class TestTheRecordIsKept:
    def test_a_lost_response_is_retried_with_the_same_bytes(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B4: the record is signed once and saved before it is posted."""
        hub: FakeHub = world["hub"]
        original = hub.handler
        calls = {"count": 0}

        def lose_the_first_response(request: httpx.Request) -> httpx.Response:
            response = original(request)
            if request.url.path.endswith("/api/v1/results"):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise httpx.ReadTimeout("the response never came back", request=request)
            return response

        monkeypatch.setattr(hub, "handler", lose_the_first_response)
        _run()

        state = load_state(world["config"])
        assert state is not None and state.published_at is not None
        stored = hub.results[(DAY, VALIDATOR.ss58_address)]
        assert json.loads(stored) == state.record

    def test_an_unreachable_hub_leaves_the_record_for_the_re_post(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hub: FakeHub = world["hub"]

        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        original = hub.handler
        monkeypatch.setattr(hub, "handler", down)
        with pytest.raises(Exception, match="transport"):
            _run()
        state = load_state(world["config"])
        assert state is not None and state.signed and state.published_at is None
        signed_bytes = state.record

        monkeypatch.setattr(hub, "handler", original)
        monkeypatch.setattr(cli.chain, "blocks_until_weights_allowed", lambda *_a, **_k: 50)
        assert _resubmit() == 0
        state = load_state(world["config"])
        assert state is not None and state.published_at is not None
        assert state.record == signed_bytes

    def test_the_record_matches_the_vector_actually_sent(self, world: dict[str, Any]) -> None:
        """B5: a miner that deregistered before the send is burn in the record too."""
        world["chain"].registered = [OWNER, MINER_A, VALIDATOR.ss58_address]
        _run()

        state = load_state(world["config"])
        assert state is not None and state.record is not None
        record = EvaluationSubmission.model_validate(state.record)
        verify_evaluation(record)
        rows = {row.hotkey: row.weight for row in record.results if row.weight}
        rows[OWNER] = rows.get(OWNER, 0) + record.burned_weight
        assert rows == state.sent_weights
        assert state.gone == [MINER_B]


class TestReposts:
    def test_submit_weights_off_stops_re_posts(
        self, world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B15: v2 kept sending the old vector with this switched off."""
        _run()
        sent = len(world["chain"].sent)
        config: Config = world["config"]
        off = config.model_copy(
            update={"validator": config.validator.model_copy(update={"submit_weights": False})}
        )
        monkeypatch.setattr(cli, "load", lambda _path: off)
        assert _resubmit() == 0
        assert len(world["chain"].sent) == sent


class TestTheStateFile:
    def _state(self) -> CycleState:
        return CycleState(
            day=DAY,
            netuid=NETUID,
            burn_hotkey=OWNER,
            scoring_version=SCORING_VERSION,
            weights=dict(WEIGHTS),
            computed_at=1,
        )

    def test_it_round_trips(self, world: dict[str, Any]) -> None:
        save_state(world["config"], self._state())
        assert load_state(world["config"]) == self._state()

    def test_a_v2_file_reads_as_a_sent_vector(self, world: dict[str, Any]) -> None:
        path = state_path(world["config"])
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "day": DAY.isoformat(),
                    "netuid": NETUID,
                    "burn_hotkey": OWNER,
                    "scoring_version": "prometheon-scoring/2.1",
                    "submitted_at": 5,
                    "weights": WEIGHTS,
                }
            )
        )
        state = load_state(world["config"])
        assert state is not None and state.sent_at == 5 and state.record is None

    def test_a_corrupt_file_is_a_typed_error_not_a_traceback(self, world: dict[str, Any]) -> None:
        """B24."""
        path = state_path(world["config"])
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        with pytest.raises(ConfigError, match="cannot be read"):
            load_state(world["config"])

    def test_a_write_leaves_no_temporary_files(self, world: dict[str, Any]) -> None:
        path = save_state(world["config"], self._state())
        assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_finishing_a_record_moves_a_departed_miners_units_to_burn() -> None:
    draft = _draft("e" * 64)
    # B left before the send, so submit_allocation sent its units to the owner.
    sent = {MINER_A: 600_000, OWNER: 400_000}
    finished = finish_record(draft, sent=sent, burn_hotkey=OWNER, uids={})
    weights = {row.hotkey: row.weight for row in finished.results}
    assert weights == {MINER_A: 600_000, MINER_B: 0}
    assert finished.burned_weight == 400_000
    assert sum(weights.values()) + finished.burned_weight == 1_000_000


def test_the_record_reproduces_a_vector_shaped_to_the_chains_limits() -> None:
    """S5: the record describes the shaped vector, padding row included."""
    draft = _draft("e" * 64)
    padding = Keypair.create_from_uri("//DurablePadding").ss58_address
    # min_allowed_weights made the cycle add a uid: 16 units, paid by A, the
    # largest entry. B and the burn are as computed.
    sent = {MINER_A: 599_984, MINER_B: 390_000, OWNER: 10_000, padding: 16}
    finished = finish_record(draft, sent=sent, burn_hotkey=OWNER, uids={padding: 9})

    weights = {row.hotkey: row.weight for row in finished.results}
    assert weights == {MINER_A: 599_984, MINER_B: 390_000, padding: 16}
    assert finished.burned_weight == 10_000
    row = next(row for row in finished.results if row.hotkey == padding)
    assert (row.uid, row.model_status, row.dataset_submitted_count) == (
        9,
        ModelStatus.NOT_COMMITTED,
        0,
    )
    # Rows plus burn are exactly what was sent, as a mirror will rebuild it.
    rebuilt = {**weights, OWNER: finished.burned_weight}
    assert rebuilt == sent


def test_a_record_cannot_name_a_padding_uid_it_does_not_know() -> None:
    draft = _draft("e" * 64)
    sent = {MINER_A: 599_984, MINER_B: 390_000, OWNER: 10_000, "5Stranger": 16}
    with pytest.raises(ScoringError, match="uid it was sent at is unknown"):
        finish_record(draft, sent=sent, burn_hotkey=OWNER, uids={})
