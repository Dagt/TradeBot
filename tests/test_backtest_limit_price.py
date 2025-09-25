"""Tests for limit price handling in the backtest engine."""

from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES
from tradingbot.utils.price import limit_price_from_close


def test_backtest_limit_price(monkeypatch):
    """Ensure that a limit signal is placed and filled at the expected price."""

    limit = 90.0

    class LimitStrategy:
        def __init__(self, risk_service=None):
            self.called = False

        def on_bar(self, _):
            if self.called:
                return None
            self.called = True
            # Emit a buy signal with a specific limit price
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

    # The engine should create exactly one order which fills at the limit price
    assert len(res["orders"]) == 1
    order = res["orders"][0]
    fill = res["fills"][0]
    fill_price = fill[3]

    assert order["place_price"] == pytest.approx(limit)
    assert order["avg_price"] == pytest.approx(limit)
    assert fill_price == pytest.approx(limit)
    # Place price, average fill price and actual fill should all match the limit
    assert fill_price == order["avg_price"] == order["place_price"]
    # Confirm the fill price differs from the bar close to ensure the limit was honoured
    assert fill_price != data["close"].iloc[-1]


def test_backtest_default_limit_price(monkeypatch):
    """Orders without an explicit limit should use the bar close as limit price."""

    class NoLimitStrategy:
        def __init__(self, risk_service=None):
            self.called = False

        def on_bar(self, _):
            if self.called:
                return None
            self.called = True
            # Emit a buy signal without specifying a limit price
            return SimpleNamespace(side="buy", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "no_limit", NoLimitStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data}, [("no_limit", "SYM")], latency=1, window=1, verbose_fills=True
    )
    res = engine.run()

    order = res["orders"][0]
    fill = res["fills"][0]
    close_price = data["close"].iloc[1]

    assert order["place_price"] == pytest.approx(close_price)
    assert order["avg_price"] == pytest.approx(close_price)
    assert fill[3] == pytest.approx(close_price)
    assert limit_price_from_close("buy", close_price, 0.0) == pytest.approx(close_price)


def test_backtest_limit_requote(monkeypatch):
    """Strategies can request a re-quote after an untouched bar and update limits."""

    initial_limit = 95.0
    requoted_limit = 99.0

    class RequoteStrategy:
        def __init__(self, risk_service=None):
            self.called = False
            self.expiry_calls = 0

        def on_bar(self, _):
            if self.called:
                return None
            self.called = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=initial_limit)

        def on_order_expiry(self, order, res):
            self.expiry_calls += 1
            assert res["status"] == "expired"
            # Request a higher limit price before re-queuing
            order.price = requoted_limit
            return "re_quote"

    monkeypatch.setitem(STRATEGIES, "limit_requote", RequoteStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4],
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 101.0, 102.0, 102.0],
            "low": [100.0, 100.0, 100.0, 98.0, 98.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000, 1000, 1000],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data}, [("limit_requote", "SYM")], latency=1, window=1, verbose_fills=True
    )
    res = engine.run()

    strat = engine.strategies[("limit_requote", "SYM")]
    assert strat.expiry_calls == 1

    order = res["orders"][0]
    fill = res["fills"][0]

    assert order["place_price"] == pytest.approx(requoted_limit)
    assert fill[3] == pytest.approx(requoted_limit)
