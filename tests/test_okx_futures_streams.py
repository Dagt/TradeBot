import json
import logging
from datetime import datetime, timezone

import pytest

from tradingbot.adapters.okx_futures import OKXFuturesAdapter
from tradingbot.adapters.base import ExchangeAdapter


@pytest.mark.asyncio
async def test_stream_order_book_ignores_zero_or_none(caplog):
    adapter = OKXFuturesAdapter.__new__(OKXFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.state = type("S", (), {"order_book": {}, "last_px": {}})()
    adapter.ws_public_url = "wss://example"

    async def fake_messages(url, sub, ping_timeout=None):
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

    with caplog.at_level(logging.WARNING):
        gen = adapter.stream_order_book("BTC/USDT", depth=1)
        ob = await gen.__anext__()
        await gen.aclose()

    assert ob["bid_px"] == [1.0]
    assert ob["ask_px"] == [3.0]
    assert "zero price" in caplog.text


@pytest.mark.asyncio
async def test_stream_bba_skips_missing_sides():
    adapter = OKXFuturesAdapter.__new__(OKXFuturesAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.ws_public_url = "wss://example"

    msg1 = json.dumps({"data": [{"bids": [], "asks": [["3", "4"]], "ts": "0"}]})
    msg2 = json.dumps({"data": [{"bids": [["1", "2"]], "asks": [["3", "4"]], "ts": "1"}]})

    async def fake_messages(url, sub, ping_timeout=None):
        yield msg1
        yield msg2

    adapter._ws_messages = fake_messages

    gen = adapter.stream_bba("BTC/USDT")
    quote = await gen.__anext__()
    await gen.aclose()

    assert quote["bid_px"] == 1.0
    assert quote["ask_px"] == 3.0

