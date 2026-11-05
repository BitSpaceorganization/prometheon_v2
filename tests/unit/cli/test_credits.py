"""`prometheon credits`: read a balance from the hub, buy credits with a stake transfer.

The chain is a stand-in that records what would have been sent, and the hub is
the fake hub. Nothing here signs or sends a real extrinsic; the local-chain run
in the docs does that.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from bittensor_wallet import Keypair

from prometheon.cli import credits as cli
from prometheon.config import (
    ChainConfig,
    ChainNetwork,
    Config,
    CreditsConfig,
    HubConfig,
    WalletConfig,
)
from prometheon.errors import ConfigError
from prometheon.hub.client import HubClient
from prometheon.hub.fake import FakeHub

pytestmark = pytest.mark.unit

NETUID = 7
COLDKEY = Keypair.create_from_uri("//CreditsCold")
HOTKEY = Keypair.create_from_uri("//CreditsHot")
TREASURY = Keypair.create_from_uri("//Treasury").ss58_address
FEE_RAO = 50 * 1_000_000_000


def _config(treasury: str = TREASURY) -> Config:
    return Config(
        chain=ChainConfig(network=ChainNetwork.LOCAL, netuid=NETUID),
        wallet=WalletConfig(name="miner", hotkey="default"),
        hub=HubConfig(base_url="https://hub.invalid"),
        credits=CreditsConfig(treasury_coldkey=treasury),
    )


class Node:
    def __init__(self, staked_rao: int) -> None:
        self.staked_rao = staked_rao
        self.transfers: list[dict[str, Any]] = []

    def get_stake(self, *, coldkey_ss58: str, hotkey_ss58: str, netuid: int) -> Any:
        return SimpleNamespace(rao=self.staked_rao)

    def transfer_stake(self, **kwargs: Any) -> Any:
        self.transfers.append(kwargs)
        return SimpleNamespace(
            success=True, message="Finalized.", error=None, extrinsic_receipt=None
        )


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Node:
    chain = Node(staked_rao=3 * FEE_RAO)
    monkeypatch.setattr(cli.chain, "connect", lambda _network: chain)

    class Balance:
        @staticmethod
        def from_rao(amount: int, netuid: int = 0) -> Any:
            return SimpleNamespace(rao=amount, netuid=netuid)

    sdk = SimpleNamespace(utils=SimpleNamespace(balance=SimpleNamespace(Balance=Balance)))
    monkeypatch.setattr(cli.chain, "_bittensor", lambda: sdk)
    monkeypatch.setattr(
        cli,
        "open_wallet",
        lambda _config: SimpleNamespace(coldkeypub=COLDKEY, hotkey=HOTKEY),
    )
    monkeypatch.delenv(cli.TREASURY_ENV, raising=False)
    return chain


def _buy(monkeypatch: pytest.MonkeyPatch, config: Config, **options: Any) -> int:
    monkeypatch.setattr(cli, "load", lambda _path: config)
    values: dict[str, Any] = {"credits": 1, "hotkey": None, "dry_run": False, "yes": True}
    values.update(options)
    return cli.cmd_buy(argparse.Namespace(config="cfg.toml", **values))


class TestBuying:
    def test_it_transfers_the_fee_to_the_treasury_on_the_same_subnet(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _buy(monkeypatch, _config(), credits=2) == 0
        (transfer,) = node.transfers
        assert transfer["destination_coldkey_ss58"] == TREASURY
        assert transfer["hotkey_ss58"] == HOTKEY.ss58_address
        assert transfer["origin_netuid"] == transfer["destination_netuid"] == NETUID
        assert transfer["amount"].rao == 2 * FEE_RAO

    def test_a_dry_run_sends_nothing(self, node: Node, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _buy(monkeypatch, _config(), dry_run=True) == 0
        assert node.transfers == []

    def test_without_confirmation_nothing_is_sent(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda _prompt: "no")
        assert _buy(monkeypatch, _config(), yes=False) == 0
        assert node.transfers == []

    def test_no_treasury_means_nothing_is_sent(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The address is published at launch; there is no built-in one to fall back to."""
        with pytest.raises(ConfigError, match="no treasury coldkey is set"):
            _buy(monkeypatch, _config(treasury=""))
        assert node.transfers == []

    def test_the_environment_can_supply_the_treasury(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(cli.TREASURY_ENV, TREASURY)
        assert _buy(monkeypatch, _config(treasury="")) == 0
        assert node.transfers[0]["destination_coldkey_ss58"] == TREASURY

    def test_two_different_treasuries_are_refused(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(cli.TREASURY_ENV, COLDKEY.ss58_address)
        with pytest.raises(ConfigError, match="different addresses"):
            _buy(monkeypatch, _config())

    def test_a_malformed_treasury_is_refused(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ConfigError, match="not an SS58 address"):
            _buy(monkeypatch, _config(treasury="not-an-address"))

    def test_too_little_stake_is_refused_before_anything_is_sent(
        self, node: Node, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.staked_rao = FEE_RAO - 1
        with pytest.raises(ConfigError, match="less than"):
            _buy(monkeypatch, _config())
        assert node.transfers == []


class TestTheBalance:
    def test_it_reads_the_public_ledger(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        hub = FakeHub(netuid=NETUID)
        hub.credits[COLDKEY.ss58_address] = {
            "coldkey": COLDKEY.ss58_address,
            "balance": 1,
            "remainder_alpha_rao": 10 * 1_000_000_000,
            "submission_fee_alpha_rao": FEE_RAO,
            "entries": [
                {
                    "kind": "deposit",
                    "block": 42,
                    "extrinsic_hash": "0xabc",
                    "alpha_rao": 60 * 1_000_000_000,
                    "credits": 1,
                }
            ],
        }
        monkeypatch.setattr(cli, "load", lambda _path: _config())
        monkeypatch.setattr(
            cli,
            "_hub",
            lambda config: HubClient(
                base_url="https://hub.invalid",
                netuid=NETUID,
                epoch_keys={},
                transport=httpx.MockTransport(hub.handler),
            ),
        )
        cli.cmd_balance(
            argparse.Namespace(config="cfg.toml", address=COLDKEY.ss58_address, limit=10)
        )
        printed = capsys.readouterr().out
        assert "credits          1" in printed
        assert "carried forward  10 alpha" in printed
        assert "block 42: deposit 60 alpha +1 credit(s) 0xabc" in printed
