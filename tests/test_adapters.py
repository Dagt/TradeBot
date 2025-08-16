import pytest
from tradingbot.adapters import (
    BybitSpotAdapter,
    BybitFuturesAdapter,
    OKXSpotAdapter,
    OKXFuturesAdapter,
)
from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.adapters.binance_spot import BinanceSpotAdapter
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.adapters.binance_spot_ws import BinanceSpotWSAdapter


async def collect_trades(adapter):
    trades = []
    async for t in adapter.stream_trades("BTCUSDT"):
        trades.append(t)
    return trades


@pytest.mark.asyncio
async def test_mock_adapter_stream_and_orders(mock_adapter, mock_trades, mock_order):
    collected = await collect_trades(mock_adapter)
    assert collected == mock_trades

    order_res = await mock_adapter.place_order(**mock_order)
    assert order_res["status"] == "placed"
    assert order_res["symbol"] == mock_order["symbol"]

    cancel = await mock_adapter.cancel_order("1")
    assert cancel["status"] == "canceled"


def test_registered_adapters_are_subclasses():
    for cls in (
        BybitSpotAdapter,
        BybitFuturesAdapter,
        OKXSpotAdapter,
        OKXFuturesAdapter,
    ):
        assert issubclass(cls, ExchangeAdapter)


class _DummyRest:
    def fetch_trades(self, symbol, limit=1):
        return [{"timestamp": 1000, "price": "1", "amount": "2", "side": "buy"}]

    def fetch_order_book(self, symbol):
        return {"timestamp": 1000, "bids": [["1", "2"]], "asks": [["3", "4"]]}


class _DummyFuturesRest:
    def fetchFundingRate(self, symbol):
        return {"rate": 0.01}

    def fetchOpenInterest(self, symbol):
        return {"oi": 100}


class _DummyDelegate:
    async def fetch_funding(self, symbol):
        return {"rate": 1}

    async def fetch_oi(self, symbol):
        return {"oi": 2}

    async def place_order(self, *a, **k):
        return {"status": "ok"}

    async def cancel_order(self, *a, **k):
        return {"status": "cancel"}


@pytest.mark.asyncio
async def test_binance_spot_rest_streams():
    adapter = BinanceSpotAdapter.__new__(BinanceSpotAdapter)
    adapter.rest = _DummyRest()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    gen = adapter.stream_trades("BTC/USDT")
    trade = await gen.__anext__()
    await gen.aclose()
    assert trade["price"] == 1.0

    gen2 = adapter.stream_order_book("BTC/USDT")
    book = await gen2.__anext__()
    await gen2.aclose()
    assert book["bids"][0][0] == 1.0


@pytest.mark.asyncio
async def test_binance_futures_rest_fetch():
    adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
    adapter.rest = _DummyFuturesRest()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    funding = await adapter.fetch_funding("BTC/USDT")
    oi = await adapter.fetch_oi("BTC/USDT")
    assert funding["rate"] == 0.01
    assert oi["oi"] == 100


@pytest.mark.asyncio
async def test_ws_delegates_to_rest():
    rest = _DummyDelegate()
    ws = BinanceSpotWSAdapter(rest=rest)
    assert await ws.fetch_funding("BTC/USDT") == {"rate": 1}
    assert await ws.fetch_oi("BTC/USDT") == {"oi": 2}
    assert await ws.place_order("BTC/USDT", "buy", "market", 1) == {"status": "ok"}
    assert await ws.cancel_order("1") == {"status": "cancel"}
