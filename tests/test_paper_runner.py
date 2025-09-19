import pytest

from tradingbot.execution.paper import PaperAdapter
from tradingbot.backtesting.engine import SlippageModel
from tradingbot.apps.api import main as api_main


@pytest.mark.asyncio
async def test_paper_runner_slippage_updates_stats():
    api_main._BOTS.clear()
    adapter = PaperAdapter(
        slippage_model=SlippageModel(volume_impact=0.0, spread_mult=1.0, pct=0.0001)
    )
    adapter.state.cash = 10_000.0
    symbol = "BTC/USDT"
    base_price = 100.0
    # Simulate a thin book where the best ask only covers part of the order and
    # additional size sits at a higher level.
    book = {
        "bid": 99.5,
        "bid_qty": 0.2,
        "bid_size": 0.2,
        "ask": 100.0,
        "ask_qty": 0.3,
        "ask_size": 0.8,
        "bids": [[99.5, 0.2]],
        "asks": [[100.0, 0.3], [101.0, 0.5]],
        "volume": 1.0,
    }
    adapter.update_last_price(symbol, base_price, book=book)
    result = await adapter.place_order(symbol, "buy", "market", 0.8, book=book)
    assert result["status"] == "filled"
    assert result["slippage_bps"] != pytest.approx(0.0)
    assert result["slippage_bps"] > 0

    api_main._BOTS[1] = {"stats": {}}
    fill_event = {
        "event": "fill",
        "qty": result["qty"],
        "fee": result.get("fee", 0.0),
        "slippage_bps": result["slippage_bps"],
        "maker": result.get("fee_type") == "maker",
        "price": result["price"],
        "order_id": result.get("order_id"),
        "pending_qty": result.get("pending_qty"),
        "filled_qty": result.get("filled_qty"),
    }
    await api_main.update_bot_stats(1, fill_event)
    stats = api_main._BOTS[1]["stats"]
    assert stats["slippage_bps"] > 0
    api_main._BOTS.clear()
