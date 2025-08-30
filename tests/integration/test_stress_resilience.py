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


@pytest.mark.integration
def test_engine_resilient_under_stress(monkeypatch):
    csv_path = "data/examples/btcusdt_3m.csv"
    monkeypatch.setitem(STRATEGIES, "buyonce", BuyOnceStrategy)
    strategies = [("buyonce", "SYM")]
    data = {"SYM": csv_path}

    base = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0),
        risk_pct=0.0,
        initial_equity=1000.0,
    )

    stressed = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0),
        stress=StressConfig(latency=2.0, spread=2.0),
        risk_pct=0.0,
        initial_equity=1000.0,
    )
    base_order = base["orders"][0]
    stress_order = stressed["orders"][0]

    assert base_order["latency"] == 1
    assert stress_order["latency"] == 2
    assert stressed["slippage"] > base["slippage"]
