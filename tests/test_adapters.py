import pytest
from datetime import datetime, timezone
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
from tradingbot.adapters.binance_futures_ws import BinanceFuturesWSAdapter


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

    def fapiPublicGetOpenInterest(self, params):
        return {"symbol": params.get("symbol"), "openInterest": "100", "time": 1000}


class _DummyDelegate:
    async def place_order(self, *a, **k):
        return {"status": "ok"}

    async def cancel_order(self, *a, **k):
        return {"status": "cancel"}


@pytest.mark.asyncio
async def test_binance_spot_rest_streams():
    adapter = BinanceSpotAdapter.__new__(BinanceSpotAdapter)
    ExchangeAdapter.__init__(adapter)
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
    assert book["bid_px"][0] == 1.0
    assert book["bid_qty"][0] == 2.0


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
    assert oi["oi"] == 100.0
    assert oi["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


@pytest.mark.asyncio
async def test_binance_futures_ws_fetch_ok():
    ws = BinanceFuturesWSAdapter()

    async def _req_funding(fn, *a, **k):
        return {"lastFundingRate": "0.05"}

    ws._request = _req_funding
    funding = await ws.fetch_funding("BTC/USDT")
    assert funding["lastFundingRate"] == "0.05"

    async def _req_oi(fn, *a, **k):
        return {"openInterest": "10"}

    ws._request = _req_oi
    oi = await ws.fetch_oi("BTC/USDT")
    assert oi["openInterest"] == "10"


@pytest.mark.asyncio
async def test_binance_futures_ws_fetch_error():
    ws = BinanceFuturesWSAdapter()

    async def _err(*a, **k):
        raise RuntimeError("boom")

    ws._request = _err
    with pytest.raises(RuntimeError):
        await ws.fetch_funding("BTC/USDT")
    with pytest.raises(RuntimeError):
        await ws.fetch_oi("BTC/USDT")


@pytest.mark.asyncio
async def test_binance_spot_ws_not_supported():
    ws = BinanceSpotWSAdapter()
    with pytest.raises(NotImplementedError):
        await ws.fetch_funding("BTC/USDT")
    with pytest.raises(NotImplementedError):
        await ws.fetch_oi("BTC/USDT")


@pytest.mark.asyncio
async def test_ws_delegates_orders_to_rest():
    rest = _DummyDelegate()
    ws = BinanceSpotWSAdapter(rest=rest)
    assert await ws.place_order("BTC/USDT", "buy", "market", 1) == {"status": "ok"}
    assert await ws.cancel_order("1") == {"status": "cancel"}


class _DummyBybitRest:
    def fetchFundingRate(self, symbol):
        return {"fundingRate": "0.01", "timestamp": 1000}

    def publicGetV5MarketOpenInterest(self, params):
        return {"result": {"list": [{"timestamp": 1000, "openInterest": "100"}]}}


class _DummyOKXRest:
    def fetchFundingRate(self, symbol):
        return {"fundingRate": "0.02", "ts": 1000}

    def publicGetPublicOpenInterest(self, params):
        return {"data": [{"ts": 1000, "oi": "200"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_cls, rest_cls, rate, oi",
    [
        (BybitSpotAdapter, _DummyBybitRest, 0.01, 100.0),
        (BybitFuturesAdapter, _DummyBybitRest, 0.01, 100.0),
        (OKXSpotAdapter, _DummyOKXRest, 0.02, 200.0),
        (OKXFuturesAdapter, _DummyOKXRest, 0.02, 200.0),
    ],
)
async def test_parsing_funding_and_oi(adapter_cls, rest_cls, rate, oi):
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.rest = rest_cls()

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    funding = await adapter.fetch_funding("BTC/USDT")
    assert funding["rate"] == rate
    assert funding["ts"] == datetime.fromtimestamp(1000, tz=timezone.utc)

    oi_res = await adapter.fetch_oi("BTC/USDT")
    assert oi_res["oi"] == oi
    assert oi_res["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)
