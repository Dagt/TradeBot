import asyncio
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


@pytest.mark.asyncio
async def test_match_book_without_volume_uses_linear_fill():
    model = SlippageModel(volume_impact=0.2, pct=0.0)
    adapter = PaperAdapter(slippage_model=model)
    adapter.state.cash = 1000.0
    symbol = "BTC/USDT"
    adapter.update_last_price(symbol, 100.0)

    order = await adapter.place_order(
        symbol,
        "buy",
        "limit",
        1.0,
        price=99.0,
        timeout=0.01,
    )
    assert order["status"] == "new"

    book = {"bid": 98.5, "ask": 99.0, "ask_size": 5.0}

    fills = adapter.update_last_price(symbol, 99.0, book=book)
    fill_events = [f for f in fills if f.get("order_id") == order["order_id"]]
    assert fill_events
    fill = fill_events[0]
    assert fill["status"] == "filled"
    assert fill["qty"] == pytest.approx(1.0)

    await asyncio.sleep(0.02)
    follow_up = adapter.update_last_price(symbol, 100.0)
    assert not any(
        evt.get("order_id") == order["order_id"] and evt.get("status") == "expired"
        for evt in follow_up
    )
