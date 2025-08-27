from types import SimpleNamespace
import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, SlippageModel
from tradingbot.strategies import STRATEGIES
from tradingbot.analysis.backtest_report import generate_report


class HalfShotStrategy:
    name = "halfshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=0.5)


@pytest.mark.integration
def test_heterogeneous_latency(monkeypatch):
    rng = pd.date_range("2021-01-01", periods=10, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "bid_size": 10.0,
            "ask_size": 10.0,
            "volume": 1000,
        }
    )
    data = {"SYM1": df.copy(), "SYM2": df.copy()}
    monkeypatch.setitem(STRATEGIES, "halfshot", HalfShotStrategy)
    strategies = [("halfshot", "SYM1", "fast"), ("halfshot", "SYM2", "slow")]
    exchange_configs = {
        "fast": {"latency": 1, "depth": 5.0},
        "slow": {"latency": 3, "depth": 5.0},
    }
    engine = EventDrivenBacktestEngine(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(),
        exchange_configs=exchange_configs,
        use_l2=True,
        initial_equity=100.0,
    )
    res = engine.run()
    orders = {o["exchange"]: o for o in res["orders"]}
    assert orders["fast"]["latency"] == 1
    assert orders["slow"]["latency"] == 3
    report = generate_report(res)
    assert pytest.approx(report["avg_latency"], rel=1e-9) == 2.0
