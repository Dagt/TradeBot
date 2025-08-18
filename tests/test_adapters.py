import json
import pytest
from datetime import datetime, timezone
from tradingbot.adapters import (
    BybitSpotAdapter,
    BybitFuturesAdapter,
    OKXSpotAdapter,
    OKXFuturesAdapter,
    DeribitAdapter,
)
from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.adapters.binance_spot import BinanceSpotAdapter
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.adapters.binance_spot_ws import BinanceSpotWSAdapter
from tradingbot.adapters.binance_futures_ws import BinanceFuturesWSAdapter
from tradingbot.adapters.binance import BinanceWSAdapter


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


@pytest.mark.asyncio
async def test_ws_adapter_reconnect(monkeypatch):
    adapter = BinanceWSAdapter()
    adapter.ping_interval = 0

    msg1 = json.dumps({"data": {"p": "1", "q": "1", "T": 0, "m": False}})
    msg2 = json.dumps({"data": {"p": "2", "q": "1", "T": 1, "m": True}})

    class DummyWS:
        def __init__(self, messages):
            self.messages = messages
            self.pings = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, msg):
            pass

        async def recv(self):
            if self.messages:
                return self.messages.pop(0)
            raise ConnectionError("closed")

        async def ping(self):
            self.pings += 1

    ws1 = DummyWS([msg1])
    ws2 = DummyWS([msg2])
    ws_iter = iter([ws1, ws2])

    monkeypatch.setattr("websockets.connect", lambda url, **k: next(ws_iter))

    sleeps: list[float] = []

    async def fake_sleep(t):
        sleeps.append(t)

    monkeypatch.setattr("tradingbot.adapters.base.asyncio.sleep", fake_sleep)

    gen = adapter.stream_trades("BTCUSDT")
    t1 = await gen.__anext__()
    t2 = await gen.__anext__()
    assert t1["price"] == 1.0
    assert t2["price"] == 2.0
    await gen.aclose()

    assert ws1.pings > 0 and ws2.pings > 0
    assert [s for s in sleeps if s >= 1][:2] == [1, 2]


def test_registered_adapters_are_subclasses():
    for cls in (
        BybitSpotAdapter,
        BybitFuturesAdapter,
        OKXSpotAdapter,
        OKXFuturesAdapter,
    ):
        assert issubclass(cls, ExchangeAdapter)


def test_adapter_testnet_selection():
    b_test = BinanceSpotAdapter(testnet=True)
    assert "testnet" in b_test.rest.urls["api"]["public"]
    b_main = BinanceSpotAdapter(testnet=False)
    assert "testnet" not in b_main.rest.urls["api"]["public"]

    bybit_test = BybitFuturesAdapter(testnet=True)
    assert "testnet" in bybit_test.rest.urls["api"]["public"]
    assert "testnet" in bybit_test.ws_public_url
    bybit_main = BybitFuturesAdapter(testnet=False)
    assert "testnet" not in bybit_main.rest.urls["api"]["public"]
    assert "testnet" not in bybit_main.ws_public_url

    okx_test = OKXFuturesAdapter(testnet=True)
    assert okx_test.rest.headers.get("x-simulated-trading") == "1"
    assert "wspap" in okx_test.ws_public_url
    okx_main = OKXFuturesAdapter(testnet=False)
    assert "x-simulated-trading" not in okx_main.rest.headers
    assert "wspap" not in okx_main.ws_public_url


class _DummyRest:
    async def fetch_trades(self, symbol, limit=1):
        return [{"timestamp": 1000, "price": "1", "amount": "2", "side": "buy"}]

    async def fetch_order_book(self, symbol):
        return {"timestamp": 1000, "bids": [["1", "2"]], "asks": [["3", "4"]]}


class _DummyFuturesRest:
    async def fetchFundingRate(self, symbol):
        return {"rate": 0.01}

    async def fapiPublicGetOpenInterest(self, params):
        return {"symbol": params.get("symbol"), "openInterest": "100", "time": 1000}

    async def public_get_premiumindex(self, params):
        return {
            "symbol": params.get("symbol"),
            "indexPrice": "100",
            "markPrice": "105",
            "time": 1000,
        }


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
    adapter.state = type("S", (), {"order_book": {}, "last_px": {}})()

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
    ExchangeAdapter.__init__(adapter)
    adapter.rest = _DummyFuturesRest()

    funding = await adapter.fetch_funding("BTC/USDT")
    basis = await adapter.fetch_basis("BTC/USDT")
    oi = await adapter.fetch_oi("BTC/USDT")
    assert funding["rate"] == 0.01
    assert basis["basis"] == 5.0
    assert isinstance(basis["ts"], datetime)
    assert oi["oi"] == 100.0
    assert oi["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


@pytest.mark.asyncio
async def test_binance_futures_rest_basis_not_supported():
    adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
    adapter.rest = type("R", (), {})()
    with pytest.raises(NotImplementedError):
        await adapter.fetch_basis("BTC/USDT")


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
    async def fetchFundingRate(self, symbol):
        return {"fundingRate": "0.01", "timestamp": 1000}

    async def publicGetV5MarketOpenInterest(self, params):
        return {"result": {"list": [{"timestamp": 1000, "openInterest": "100"}]}}

    async def publicGetV5MarketPremiumIndexPrice(self, params):
        return {
            "result": {
                "list": [
                    {"timestamp": 1000, "markPrice": "105", "indexPrice": "100"}
                ]
            }
        }


class _DummyOKXRest:
    async def fetchFundingRate(self, symbol):
        return {"fundingRate": "0.02", "ts": 1000}

    async def publicGetPublicOpenInterest(self, params):
        return {"data": [{"ts": 1000, "oi": "200"}]}

    async def fetchTicker(self, symbol):
        return {"timestamp": 1000, "markPrice": "105", "indexPrice": "100"}


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
    ExchangeAdapter.__init__(adapter)
    adapter.rest = rest_cls()

    funding = await adapter.fetch_funding("BTC/USDT")
    assert funding["rate"] == rate
    assert funding["ts"] == datetime.fromtimestamp(1000, tz=timezone.utc)

    oi_res = await adapter.fetch_oi("BTC/USDT")
    assert oi_res["oi"] == oi
    assert oi_res["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter_cls, rest_cls",
    [
        (BybitSpotAdapter, _DummyBybitRest),
        (BybitFuturesAdapter, _DummyBybitRest),
        (OKXSpotAdapter, _DummyOKXRest),
        (OKXFuturesAdapter, _DummyOKXRest),
    ],
)
async def test_fetch_basis(adapter_cls, rest_cls):
    adapter = adapter_cls.__new__(adapter_cls)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = rest_cls()

    basis = await adapter.fetch_basis("BTC/USDT")
    assert basis["basis"] == 5.0
    assert basis["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


class _DummyDeribitRest:
    async def public_get_get_funding_rate(self, params):
        return {"result": {"funding_rate": "0.01", "timestamp": 1000}}

    async def public_get_ticker(self, params):
        return {"result": {"index_price": "100", "mark_price": "105", "timestamp": 1000}}

    async def public_get_get_open_interest(self, params):
        return {"result": {"open_interest": "200", "timestamp": 1000}}


@pytest.mark.asyncio
async def test_deribit_fetch_methods():
    adapter = DeribitAdapter.__new__(DeribitAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = _DummyDeribitRest()

    funding = await adapter.fetch_funding("BTC-PERPETUAL")
    assert funding["rate"] == 0.01
    basis = await adapter.fetch_basis("BTC-PERPETUAL")
    assert basis["basis"] == 5.0
    oi = await adapter.fetch_oi("BTC-PERPETUAL")
    assert oi["oi"] == 200.0
    assert oi["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


@pytest.mark.asyncio
async def test_request_and_close_async_rest():
    class DummyAdapter(ExchangeAdapter):
        name = "dummy"

        async def stream_trades(self, symbol: str):
            if False:
                yield

        async def place_order(self, *a, **k):
            return {}

        async def cancel_order(self, *a, **k):
            return {}

    adapter = DummyAdapter()

    async def _coro(x):
        return x + 1

    assert await adapter._request(_coro, 41) == 42

    class _R:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    adapter.rest = _R()
    await adapter.close()
    assert adapter.rest.closed
