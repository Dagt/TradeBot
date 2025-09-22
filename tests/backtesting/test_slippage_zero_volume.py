import math
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, SlippageModel
from tradingbot.strategies import STRATEGIES


def test_slippage_handles_zero_volume(monkeypatch):
    class ZeroVolumeStrategy:
        def __init__(self, risk_service=None):
            self._sent = False

        def on_bar(self, _):
            if self._sent:
                return None
            self._sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=101.0)

    monkeypatch.setitem(STRATEGIES, "zero_vol_guard", ZeroVolumeStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 0.0, 1000.0],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("zero_vol_guard", "SYM")],
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=10.0, pct=0.0),
        verbose_fills=True,
    )
    result = engine.run()

    order_fills = [f for f in result["fills"] if f[1] == "order"]
    assert len(order_fills) == 1
    order_fill = order_fills[0]

    fill_price = order_fill[3]
    fill_qty = order_fill[4]

    assert math.isfinite(fill_price)
    assert math.isfinite(fill_qty)
    assert fill_qty > 0
    assert order_fill[0] == data["timestamp"].iloc[-1]

    base_price = data["close"].iloc[-1]
    assert fill_price > base_price
    assert fill_price - base_price < 1.0

    order_summary = result["orders"][0]
    expected_slip = (fill_price - order_summary["place_price"]) * fill_qty

    assert math.isfinite(result["slippage"])
    assert result["slippage"] == pytest.approx(expected_slip)
    assert abs(result["slippage"]) > 0.0
    assert abs(result["slippage"]) < 100.0
