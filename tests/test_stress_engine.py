import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import SlippageModel, StressConfig, run_backtest_csv
from tradingbot.strategies import STRATEGIES


class BuyOnceStrategy:
    name = "buyonce"

    def __init__(self):
        self.sent = False

    def on_bar(self, context):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=1.0)


def _make_csv(tmp_path):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_latency_and_spread_stress(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setitem(STRATEGIES, "buyonce", BuyOnceStrategy)
    strategies = [("buyonce", "SYM")]
    data = {"SYM": str(csv_path)}

    base = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0),
    )

    stressed = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0),
        stress=StressConfig(latency=2.0, spread=2.0),
    )

    base_order = base["orders"][0]
    stress_order = stressed["orders"][0]

    assert base_order["latency"] == 1
    assert stress_order["latency"] == 2

    base_slip = base_order["avg_price"] - base_order["place_price"]
    stress_slip = stress_order["avg_price"] - stress_order["place_price"]
    assert pytest.approx(stress_slip, rel=1e-9) == base_slip * 2
