import pytest
from tradingbot.apps.api.main import strategy_schema


def test_triangular_arb_schema_returns_params():
    res = strategy_schema("triangular_arb")
    names = [p["name"] for p in res["params"]]
    assert "taker_fee_bps" in names


def test_cross_arbitrage_schema_returns_params():
    res = strategy_schema("cross_arbitrage")
    names = [p["name"] for p in res["params"]]
    assert "threshold" in names
