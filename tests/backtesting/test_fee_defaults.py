from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


def _register_post_only_strategy(monkeypatch, name, limit_price):
    class PostOnlyStrategy:
        def __init__(self, risk_service=None):  # pragma: no cover - signature compatibility
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(
                side="buy",
                strength=1.0,
                limit_price=limit_price,
                post_only=True,
            )

    monkeypatch.setitem(STRATEGIES, name, PostOnlyStrategy)


def _base_frame():
    return pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )


def test_post_only_fill_uses_maker_fee(monkeypatch):
    strategy_name = "post_only_fee_check"
    limit_price = 99.0
    _register_post_only_strategy(monkeypatch, strategy_name, limit_price)

    engine = EventDrivenBacktestEngine(
        {"SYM": _base_frame()},
        [(strategy_name, "SYM", "okx_spot")],
        latency=1,
        window=1,
        verbose_fills=True,
        fee_bps=None,
        exchange_configs={
            "okx_spot": {"maker_fee_bps": 1, "taker_fee_bps": 8},
        },
    )

    result = engine.run()

    assert engine.default_fee.maker_fee == pytest.approx(0.0001)
    assert engine.default_fee.taker_fee == pytest.approx(0.0008)
    assert engine.default_maker_bps == pytest.approx(1.0)
    assert engine.default_taker_bps == pytest.approx(8.0)

    order_fills = [fill for fill in result["fills"] if fill[1] == "order"]
    assert len(order_fills) == 1

    (
        _,
        _,
        side,
        price,
        qty,
        _,
        _,
        exchange,
        fee_cost,
        *_
    ) = order_fills[0]

    assert side == "buy"
    assert exchange == "okx_spot"

    maker_rate = engine.exchange_fees["okx_spot"].maker_fee
    expected_fee = price * qty * maker_rate
    assert fee_cost == pytest.approx(expected_fee)


def test_default_fee_falls_back_to_realistic_rates_when_missing_config(monkeypatch):
    strategy_name = "post_only_fee_missing_config"
    limit_price = 99.0
    _register_post_only_strategy(monkeypatch, strategy_name, limit_price)

    engine = EventDrivenBacktestEngine(
        {"SYM": _base_frame()},
        [(strategy_name, "SYM", "unconfigured_spot")],
        latency=1,
        window=1,
        verbose_fills=True,
        fee_bps=None,
        exchange_configs={},
    )

    assert engine.default_maker_bps == pytest.approx(1.0)
    assert engine.default_taker_bps == pytest.approx(8.0)
