import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


def test_backtest_limit_price(monkeypatch):
    limit = 90.0

    class LimitStrategy:
        def __init__(self, risk_service=None):
            self.called = False

        def on_bar(self, _):
            if self.called:
                return None
            self.called = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=limit)

    monkeypatch.setitem(STRATEGIES, "limit", LimitStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 90.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data}, [("limit", "SYM")], latency=1, window=1, verbose_fills=True
    )
    res = engine.run()

    assert len(res["orders"]) == 1
    order = res["orders"][0]
    assert order["place_price"] == pytest.approx(limit)
    assert order["avg_price"] == pytest.approx(limit)

    fill = res["fills"][0]
    assert fill[1] == "order"
    assert fill[3] == pytest.approx(limit)
    assert fill[3] != data["close"].iloc[2]
