import json
from datetime import datetime, timezone

import pytest

from tradingbot.adapters.okx_futures import OKXFuturesAdapter
from tradingbot.adapters.base import ExchangeAdapter


@pytest.mark.asyncio
async def test_stream_order_book_ignores_zero_or_none():
    adapter = OKXFuturesAdapter.__new__(OKXFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.state = type("S", (), {"order_book": {}, "last_px": {}})()
    adapter.ws_public_url = "wss://example"

    async def fake_messages(url, sub):
        yield json.dumps(
            {"data": [{"bidPx": None, "askPx": "3", "bidSz": "1", "askSz": "1", "ts": "0"}]}
        )
        yield json.dumps(
            {"data": [{"bidPx": "1", "askPx": "0", "bidSz": "1", "askSz": "1", "ts": "1"}]}
        )
        yield json.dumps(
            {"data": [{"bidPx": "1", "bidSz": "2", "askPx": "3", "askSz": "4", "ts": "2"}]}
        )

    adapter._ws_messages = fake_messages

    gen = adapter.stream_order_book("BTC/USDT", depth=1)
    ob = await gen.__anext__()
    await gen.aclose()

    assert ob["bid_px"] == [1.0]
    assert ob["ask_px"] == [3.0]


@pytest.mark.asyncio
async def test_stream_bba_skips_missing_sides():
    adapter = OKXFuturesAdapter.__new__(OKXFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    ts = datetime.fromtimestamp(0, tz=timezone.utc)

    async def fake_order_book(symbol, depth):
        yield {"ts": ts, "bid_px": [], "bid_qty": [], "ask_px": [3.0], "ask_qty": [4.0]}
        yield {"ts": ts, "bid_px": [1.0], "bid_qty": [2.0], "ask_px": [3.0], "ask_qty": [4.0]}

    adapter.stream_order_book = fake_order_book

    gen = adapter.stream_bba("BTC/USDT")
    quote = await gen.__anext__()
    await gen.aclose()

    assert quote["bid_px"] == 1.0
    assert quote["ask_px"] == 3.0

