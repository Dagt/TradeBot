import math

import pytest

from tradingbot.execution.paper import PaperAdapter
from tradingbot.backtesting.engine import SlippageModel


@pytest.mark.parametrize("book", [
    {"ask": 101.0, "ask_size": 10.0, "bid": 99.0, "bid_size": 15.0},
    {"ask": 101.0, "ask_size": 10.0, "bid": 99.0, "bid_size": 15.0, "asks": []},
])
def test_apply_slippage_without_volume_uses_fallback(book):
    adapter = PaperAdapter(slippage_model=SlippageModel(volume_impact=0.2, pct=0.0))

    px, slip_bps = adapter._apply_slippage("BTCUSDT", "buy", 1.0, 100.0, book)

    assert px == pytest.approx(100.0)
    assert slip_bps == pytest.approx(0.0)


def test_apply_slippage_with_volume_uses_model():
    adapter = PaperAdapter(slippage_model=SlippageModel(volume_impact=0.2, pct=0.0))

    book = {
        "ask": 101.0,
        "ask_size": 10.0,
        "bid": 99.0,
        "bid_size": 15.0,
        "volume": 5_000.0,
    }

    px, slip_bps = adapter._apply_slippage("BTCUSDT", "buy", 5.0, 100.0, book)

    assert math.isfinite(px)
    assert math.isfinite(slip_bps)
    assert px > 100.0
    assert slip_bps > 0.0
