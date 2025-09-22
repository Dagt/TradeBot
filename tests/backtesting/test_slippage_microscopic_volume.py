from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, SlippageModel
from tradingbot.strategies import STRATEGIES


def test_microscopic_volume_skips_fill_and_slippage(monkeypatch):
    class MicroVolumeStrategy:
        def __init__(self, risk_service=None):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=101.0)

    monkeypatch.setitem(STRATEGIES, "micro_volume_guard", MicroVolumeStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1e-12, 1e-12],
        }
    )

    slippage = SlippageModel(volume_impact=10.0, pct=0.0)
    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("micro_volume_guard", "SYM")],
        latency=1,
        window=1,
        slippage=slippage,
        verbose_fills=True,
    )
    result = engine.run()

    order_fills = [f for f in result["fills"] if f[1] == "order"]
    assert order_fills == []
    assert result["slippage"] == pytest.approx(0.0)

    adj_price = engine.slippage.adjust("buy", 1.0, 100.0, {"volume": 1e-12})
    assert adj_price == pytest.approx(100.0)

    fill_price, fill_qty, _ = engine.slippage.fill(
        "buy", 1.0, 100.0, {"volume": 1e-12}, 0.0, True
    )
    assert fill_qty == pytest.approx(0.0)
    assert fill_price == pytest.approx(100.0)


def test_healthy_volume_fill_matches_original_behaviour(monkeypatch):
    class HealthyVolumeStrategy:
        def __init__(self, risk_service=None):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=101.0)

    monkeypatch.setitem(STRATEGIES, "healthy_volume_guard", HealthyVolumeStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0],
        }
    )

    slippage = SlippageModel(volume_impact=10.0, pct=0.0)
    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("healthy_volume_guard", "SYM")],
        latency=1,
        window=1,
        slippage=slippage,
        verbose_fills=True,
    )
    result = engine.run()

    order_fills = [f for f in result["fills"] if f[1] == "order"]
    assert len(order_fills) == 1
    fill_timestamp, _kind, _side, fill_price, fill_qty, *_ = order_fills[0]
    assert fill_qty > 0

    bar_volume = float(data.loc[data["timestamp"] == fill_timestamp, "volume"].iloc[0])
    base_price = float(data.loc[data["timestamp"] == fill_timestamp, "close"].iloc[0])
    expected_price = base_price + slippage.volume_impact * fill_qty / bar_volume
    assert fill_price == pytest.approx(expected_price)

    order_summary = next(
        order
        for order in result["orders"]
        if order["strategy"] == "healthy_volume_guard" and order["side"] == "buy"
    )
    expected_slippage = (fill_price - order_summary["place_price"]) * fill_qty
    assert result["slippage"] == pytest.approx(expected_slippage)
