import asyncio
import os
import pytest

from tradingbot.adapters.bybit_ws import BybitWSAdapter


requires_net = pytest.mark.skipif(
    not os.getenv("BYBIT_WS_TEST"), reason="requires network access to Bybit"
)


@requires_net
@pytest.mark.asyncio
async def test_stream_order_book_snapshot():
    adapter = BybitWSAdapter(testnet=True)
    stream = adapter.stream_order_book("BTC/USDT", depth=1)
    try:
        book = await asyncio.wait_for(stream.__anext__(), timeout=20)
    finally:
        await stream.aclose()
    assert book["bids"] and book["asks"]


@requires_net
@pytest.mark.asyncio
async def test_stream_book_delta_update():
    adapter = BybitWSAdapter(testnet=True)
    stream = adapter.stream_book_delta("BTC/USDT", depth=1)
    try:
        delta = await asyncio.wait_for(stream.__anext__(), timeout=20)
    finally:
        await stream.aclose()
    # Ensure keys exist; lists may be empty if no changes but message arrives
    assert "bid_px" in delta and "ask_px" in delta
