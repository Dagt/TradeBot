import pytest

from tradingbot.adapters.binance_futures_ws import BinanceFuturesWSAdapter
from tradingbot.config import settings


@pytest.mark.asyncio
async def test_place_and_cancel_order_on_testnet():
    if not (settings.binance_futures_api_key and settings.binance_futures_api_secret):
        pytest.skip("Binance Futures testnet credentials required")
    adapter = BinanceFuturesWSAdapter(testnet=settings.binance_futures_testnet)
    order = await adapter.place_order(
        "BTC/USDT", "buy", "LIMIT", 0.001, price=100.0
    )
    assert order.get("ext_order_id")
    cancel = await adapter.cancel_order(order["ext_order_id"], symbol="BTC/USDT")
    assert str(cancel.get("orderId")) == str(order["ext_order_id"])
