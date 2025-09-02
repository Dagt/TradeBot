import pytest
from tradingbot.apps.api.main import strategy_schema
from tradingbot.strategies import STRATEGIES


def test_triangular_arb_schema_returns_params():
    res = strategy_schema("triangular_arb")
    names = [p["name"] for p in res["params"]]
    assert "taker_fee_bps" in names


def test_cross_arbitrage_schema_returns_params():
    res = strategy_schema("cross_arbitrage")
    names = [p["name"] for p in res["params"]]
    assert "threshold" in names


def test_strategy_schema_filters_internal_params():
    internal = {
        "config_path",
        "risk_service",
        "spot_exchange",
        "perp_exchange",
        "persist_pg",
        "rebalance_assets",
        "rebalance_threshold",
        "latency",
    }
    for name in list(STRATEGIES) + ["cross_arbitrage"]:
        res = strategy_schema(name)
        names = {p["name"] for p in res["params"]}
        assert not (internal & names), f"{name} exposes internal params: {internal & names}"
