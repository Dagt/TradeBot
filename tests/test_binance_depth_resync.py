import asyncio
import json
import pytest

from tradingbot.adapters.binance_spot_ws import BinanceSpotWSAdapter
from tradingbot.adapters.binance_futures_ws import BinanceFuturesWSAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [BinanceSpotWSAdapter, BinanceFuturesWSAdapter])
async def test_depth_update_gap_resync(monkeypatch, cls):
    adapter = cls()
    calls = {"ws": 0, "rest": 0}

    first_msgs = [
        json.dumps({"data": {"b": [["1", "1"]], "a": [["2", "1"]], "pu": 9, "u": 10, "T": 0}}),
        json.dumps({"data": {"b": [["1", "1"]], "a": [["2", "1"]], "pu": 12, "u": 13, "T": 1}}),
    ]
    second_msgs = [
        json.dumps({"data": {"b": [["1", "1"]], "a": [["2", "1"]], "pu": 20, "u": 21, "T": 2}})
    ]

    async def fake_ws_messages(url, subscribe=None, ping_timeout=None):
        calls["ws"] += 1
        msgs = first_msgs if calls["ws"] == 1 else second_msgs
        for m in msgs:
            yield m
        while True:
            await asyncio.sleep(0.01)

    class DummyRest:
        async def fetch_order_book(self, symbol, depth=10):
            calls["rest"] += 1
            return {"lastUpdateId": 20, "bids": [["1", "1"]], "asks": [["2", "1"]]}

    adapter._ws_messages = fake_ws_messages
    adapter.rest = DummyRest()

    gen = adapter.stream_book_delta("BTC/USDT", depth=10)
    first = await gen.__anext__()
    second = await gen.__anext__()
    third = await gen.__anext__()
    await gen.aclose()

    assert calls["ws"] >= 2
    assert calls["rest"] == 1
    assert first["bid_px"] == [1.0]
    assert second["bid_px"] == [1.0]
    assert third["bid_px"] == []
