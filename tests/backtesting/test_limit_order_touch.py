from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


def _register_strategy(monkeypatch, name, limit_price):
    class SingleLimitStrategy:
        def __init__(self, risk_service=None):  # pragma: no cover - signature compatibility
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=limit_price)

    monkeypatch.setitem(STRATEGIES, name, SingleLimitStrategy)


def _build_engine(data, strategy_name):
    return EventDrivenBacktestEngine(
        {"SYM": data},
        [(strategy_name, "SYM")],
        latency=1,
        window=1,
        verbose_fills=True,
    )


def _base_frame(low_values):
    return pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": low_values,
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )


def test_limit_order_waits_until_bar_touches_price(monkeypatch):
    limit_price = 99.0
    _register_strategy(monkeypatch, "limit_skip", limit_price)
    data = _base_frame([100.0, 100.0, 100.0])

    engine = _build_engine(data, "limit_skip")
    result = engine.run()

    assert result["fill_count"] == 0
    assert result["fills"] == []
    assert result["cancel_count"] == 1


def test_limit_order_fills_when_bar_touches_price(monkeypatch):
    limit_price = 99.0
    _register_strategy(monkeypatch, "limit_fill", limit_price)
    data = _base_frame([100.0, 100.0, 98.0])

    engine = _build_engine(data, "limit_fill")
    result = engine.run()

    assert result["fill_count"] >= 1

    order_fills = [fill for fill in result["fills"] if fill[1] == "order"]
    assert len(order_fills) == 1

    (
        timestamp,
        reason,
        side,
        price,
        qty,
        strategy,
        symbol,
        exchange,
        fee_cost,
        slippage_pnl,
        realized_pnl,
        realized_total,
        equity_after,
    ) = order_fills[0]

    assert side == "buy"
    assert price == pytest.approx(limit_price)
    expected_fee = price * qty * engine.default_fee.taker_fee
    assert fee_cost == pytest.approx(expected_fee)
    assert slippage_pnl == pytest.approx(0.0)
    assert realized_pnl == pytest.approx(-expected_fee)
    assert realized_total == pytest.approx(-expected_fee)
