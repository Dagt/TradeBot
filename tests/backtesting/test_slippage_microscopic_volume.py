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


def test_fill_respects_max_bar_participation():
    model = SlippageModel(
        volume_impact=0.0,
        pct=0.0,
        max_bar_participation=0.25,
        min_bar_liquidity=0.0,
    )

    bar = {"volume": 100.0}
    qty = 80.0
    queue_pos = 0.0
    fills = []

    for _ in range(10):
        _, fill_qty, queue_pos = model.fill(
            "buy", qty, 100.0, bar, queue_pos=queue_pos, partial=True
        )
        fills.append(fill_qty)
        qty -= fill_qty
        if qty <= 1e-9:
            break
        # avoid infinite loop in case of regression
        assert fill_qty > 0

    assert qty <= 1e-9
    assert fills[:3] == pytest.approx([25.0, 25.0, 25.0])
    assert fills[3] == pytest.approx(5.0)


def test_fill_respects_min_bar_liquidity():
    model = SlippageModel(
        volume_impact=0.0,
        pct=0.0,
        max_bar_participation=0.1,
        min_bar_liquidity=2.0,
    )

    bar = {"volume": 10.0}
    price, fill_qty, queue_pos = model.fill(
        "buy", 5.0, 100.0, bar, queue_pos=0.0, partial=True
    )

    assert fill_qty == pytest.approx(0.0)
    assert price == pytest.approx(100.0)
    assert queue_pos == pytest.approx(0.0)


def test_engine_partial_fills_respect_participation(monkeypatch):
    class LargeOrderStrategy:
        def __init__(self, risk_service=None):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=105.0)

    monkeypatch.setitem(STRATEGIES, "large_participation", LargeOrderStrategy)
    monkeypatch.setattr(
        "tradingbot.backtesting.engine.RiskService.calc_position_size",
        lambda self, strength, price, *args, **kwargs: 100.0,
    )

    data = pd.DataFrame(
        {
            "timestamp": list(range(8)),
            "open": [100.0] * 8,
            "high": [100.0] * 8,
            "low": [100.0] * 8,
            "close": [100.0] * 8,
            "volume": [100.0] * 8,
        }
    )

    slippage = SlippageModel(
        volume_impact=0.0,
        pct=0.0,
        max_bar_participation=0.25,
    )
    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("large_participation", "SYM")],
        latency=1,
        window=1,
        slippage=slippage,
        verbose_fills=True,
    )
    result = engine.run()

    order_fills = [f for f in result["fills"] if f[1] == "order"]
    fill_qtys = [f[4] for f in order_fills]

    # Expect four fills of 25 units each for the 100 unit order
    assert len(fill_qtys) >= 4
    assert fill_qtys[:3] == pytest.approx([25.0, 25.0, 25.0])
    assert fill_qtys[3] == pytest.approx(25.0)
