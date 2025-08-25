import os

import pytest
from typer.testing import CliRunner

from tradingbot.cli.main import app


def test_kaiko_connector_requires_api_key(monkeypatch):
    monkeypatch.delenv("KAIKO_API_KEY", raising=False)
    from tradingbot.connectors.kaiko import KaikoConnector

    with pytest.raises(ValueError, match="KAIKO_API_KEY missing"):
        KaikoConnector()


def test_coinapi_connector_requires_api_key(monkeypatch):
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    from tradingbot.connectors.coinapi import CoinAPIConnector

    with pytest.raises(ValueError, match="COINAPI_KEY missing"):
        CoinAPIConnector()


def test_cli_ingest_historical_reports_missing_key(monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("KAIKO_API_KEY", raising=False)
    result = runner.invoke(
        app, ["ingest-historical", "kaiko", "BTC-USD", "--exchange", "binance_spot"]
    )
    assert result.exit_code != 0
    assert "KAIKO_API_KEY missing" in result.stdout


def test_cli_ingest_historical_coinapi_missing_key(monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("COINAPI_KEY", raising=False)
    result = runner.invoke(app, ["ingest-historical", "coinapi", "BTCUSD"])
    assert result.exit_code != 0
    assert "COINAPI_KEY missing" in result.stdout
