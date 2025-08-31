import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES
from tradingbot.risk.service import RiskService


def test_account_cash_updates_after_each_fill(monkeypatch):
    # create simple price data
    rng = pd.date_range("2021-01-01", periods=4, freq="T")
    df = pd.DataFrame({
        "timestamp": rng.view("int64") // 10**9,
        "open": 100.0,
        "high": 100.0,
        "low": 100.0,
        "close": 100.0,
        "volume": 1000,
    })
    data = {"SYM": df}

    class Signal(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

    class BuySell:
        def __init__(self):
            self.i = 0

        def on_bar(self, bar):
            self.i += 1
            if self.i == 1:
                return Signal(side="buy", strength=1.0)
            if self.i == 2:
                return Signal(side="sell", strength=1.0)
            return Signal(side="flat", strength=0.0)

    monkeypatch.setitem(STRATEGIES, "buysell", BuySell)
    strategies = [("buysell", "SYM")]
    engine = EventDrivenBacktestEngine(
        data, strategies, latency=1, window=1, fee_bps=0.0, initial_equity=1000.0
    )

    balances = []
    original_on_fill = RiskService.on_fill

    def record_balance(self, symbol, side, qty, price, venue=None):
        original_on_fill(self, symbol, side, qty, price, venue=venue)
        self.account.open_orders.clear()
        balances.append(self.account.get_available_balance())

    monkeypatch.setattr(RiskService, "on_fill", record_balance)

    engine.run()

    assert len(balances) == 2
    assert balances[0] == pytest.approx(0.0)
    assert balances[1] == pytest.approx(1000.0)
