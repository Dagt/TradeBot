import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


class DummyStrategy:
    name = "dummy"

    def on_bar(self, bar):
        return SimpleNamespace(side="buy", strength=1.0)


@pytest.mark.integration
def test_event_engine_runs(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="min")
    price = np.linspace(100, 104, num=5)
    df = pd.DataFrame({
        "timestamp": rng.view("int64") // 10**9,
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
        "volume": 1000,
    })
    data = {"SYM": df}
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    engine = EventDrivenBacktestEngine(data, [("dummy", "SYM")], latency=1, window=1)
    res = engine.run()
    assert "equity" in res
    assert len(res["fills"]) > 0


def test_event_engine_single_symbol_cov(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=6, freq="min")
    price = np.linspace(100, 105, num=6)
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        }
    )
    data = {"SYM": df}
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    engine = EventDrivenBacktestEngine(data, [("dummy", "SYM")], latency=1, window=3)
    res = engine.run()
    assert "equity" in res
    assert len(res["fills"]) > 0


class OneShotStrategy:
    name = "oneshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=1.0)


class HalfShotStrategy:
    name = "halfshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=0.5)


def test_stop_loss_triggers_close(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="min")
    price = [100.0, 100.0, 100.0, 80.0, 80.0]
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": [p + 0.5 for p in price],
            "low": [p - 0.5 for p in price],
            "close": price,
            "volume": 1000,
        }
    )
    data = {"SYM": df}
    monkeypatch.setitem(STRATEGIES, "halfshot", HalfShotStrategy)
    engine = EventDrivenBacktestEngine(
        data,
        [("halfshot", "SYM")],
        latency=1,
        window=1,
        risk_pct=0.1,
        initial_equity=1000,
    )
    res = engine.run()
    assert len(res["orders"]) == 2
    assert res["orders"][0]["side"] == "buy"
    assert res["orders"][1]["side"] == "sell"
    entry_price = res["fills"][0][2]
    exit_price = res["fills"][1][2]
    assert exit_price <= entry_price * (1 - 0.1)
    assert res["orders"][1]["filled"] == res["orders"][0]["qty"]


def test_equity_loss_capped_by_risk_pct(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="min")
    price = [100.0, 100.0, 100.0, 94.0, 95.0]
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": [p + 0.5 for p in price],
            "low": [p - 0.5 for p in price],
            "close": price,
            "volume": 1000,
        }
    )
    data = {"SYM": df}
    monkeypatch.setitem(STRATEGIES, "oneshot", OneShotStrategy)
    initial_equity = 1000.0
    risk_pct = 0.05
    engine = EventDrivenBacktestEngine(
        data,
        [("oneshot", "SYM")],
        latency=1,
        window=1,
        risk_pct=risk_pct,
        initial_equity=initial_equity,
    )
    res = engine.run()
    orders = res["orders"]
    notional = orders[0]["avg_price"] * orders[0]["filled"]
    loss = initial_equity - res["equity"]
    assert loss <= notional * risk_pct + 1e-9
