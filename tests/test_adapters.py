import asyncio
import json
import logging

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


@pytest.mark.asyncio
async def test_ws_messages_handle_ping_and_heartbeat(monkeypatch):
    class DummyAdapter(ExchangeAdapter):
        name = "dummy"

        async def stream_trades(self, symbol: str):  # pragma: no cover - not used
            raise NotImplementedError

        async def place_order(
            self,
            symbol: str,
            side: str,
            type_: str,
            qty: float,
            price: float | None = None,
            post_only: bool = False,
            time_in_force: str | None = None,
            reduce_only: bool = False,
        ) -> dict:
            return {}

        async def cancel_order(self, order_id: str) -> dict:
            return {}

    ping_message = json.dumps({"event": "ping", "args": [{"channel": "trades"}]})
    heartbeat_message = json.dumps({"method": "heartbeat"})
    payload_message = json.dumps({"foo": "bar"})

    class DummyWS:
        def __init__(self, messages):
            self.messages = list(messages)
            self.sent: list[str] = []
            self.pings = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, msg):
            self.sent.append(msg)

        async def recv(self):
            if self.messages:
                return self.messages.pop(0)
            raise ConnectionError("closed")

        async def ping(self):
            self.pings += 1

    ws = DummyWS([ping_message, heartbeat_message, payload_message])
    connect_calls = {"count": 0}

    def fake_connect(*args, **kwargs):
        connect_calls["count"] += 1
        return ws

    monkeypatch.setattr("websockets.connect", fake_connect)

    class DummyMetric:
        def __init__(self):
            self.count = 0

        def labels(self, **kwargs):
            return self

        def inc(self):
            self.count += 1

    reconnect_metric = DummyMetric()
    failure_metric = DummyMetric()
    monkeypatch.setattr("tradingbot.adapters.base.WS_RECONNECTS", reconnect_metric)
    monkeypatch.setattr("tradingbot.adapters.base.WS_FAILURES", failure_metric)

    adapter = DummyAdapter()
    adapter.ping_interval = 3600

    gen = adapter._ws_messages(
        "wss://example",
        subscribe=json.dumps({"op": "subscribe", "args": [{"channel": "trades"}]})
    )

    raw = await gen.__anext__()
    assert json.loads(raw) == {"foo": "bar"}
    await gen.aclose()

    assert connect_calls["count"] == 1
    assert len(ws.sent) >= 3
    pong = json.loads(ws.sent[1])
    assert pong["op"] == "pong"
    assert pong.get("args") == [{"channel": "trades"}]
    if "event" in pong:
        assert pong["event"] == "pong"
    heartbeat = json.loads(ws.sent[2])
    assert heartbeat == {
        "jsonrpc": "2.0",
        "id": 0,
        "method": "public/respond-heartbeat",
    }
    assert reconnect_metric.count == 0
    assert failure_metric.count == 0


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


class _DummyBinanceSpotRest:
    async def fapiPublicGetFundingRate(self, params):
        return [{"fundingRate": "0.01", "fundingTime": 1000}]

    async def fapiPublicGetPremiumIndex(self, params):
        return {"markPrice": "105", "indexPrice": "100", "time": 1000}

    async def fapiPublicGetOpenInterest(self, params):
        return {"openInterest": "100", "time": 1000}


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

    gen3 = adapter.stream_bba("BTC/USDT")
    bba = await gen3.__anext__()
    await gen3.aclose()
    assert bba["bid_px"] == 1.0
    assert bba["bid_qty"] == 2.0
    assert bba["ask_px"] == 3.0
    assert bba["ask_qty"] == 4.0

    gen4 = adapter.stream_book_delta("BTC/USDT")
    delta = await gen4.__anext__()
    await gen4.aclose()
    assert delta["bid_px"][0] == 1.0
    assert delta["ask_px"][0] == 3.0


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

    sf_gen = adapter.stream_funding("BTC/USDT")
    sf = await sf_gen.__anext__()
    await sf_gen.aclose()
    assert sf["rate"] == 0.01

    oi_gen = adapter.stream_open_interest("BTC/USDT")
    soi = await oi_gen.__anext__()
    await oi_gen.aclose()
    assert soi["oi"] == 100.0


@pytest.mark.asyncio
async def test_binance_futures_rest_basis_not_supported():
    adapter = BinanceFuturesAdapter.__new__(BinanceFuturesAdapter)
    adapter.rest = type("R", (), {})()
    with pytest.raises(NotImplementedError):
        await adapter.fetch_basis("BTC/USDT")


@pytest.mark.asyncio
async def test_binance_spot_fetch_methods():
    adapter = BinanceSpotAdapter.__new__(BinanceSpotAdapter)
    ExchangeAdapter.__init__(adapter)
    adapter.rest = _DummyBinanceSpotRest()

    funding = await adapter.fetch_funding("BTC/USDT")
    basis = await adapter.fetch_basis("BTC/USDT")
    oi = await adapter.fetch_oi("BTC/USDT")

    assert funding["rate"] == 0.01
    assert funding["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)
    assert basis["basis"] == 5.0
    assert basis["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)
    assert oi["oi"] == 100.0
    assert oi["ts"] == datetime.fromtimestamp(1, tz=timezone.utc)


@pytest.mark.asyncio
async def test_binance_spot_fetch_methods_not_supported():
    adapter = BinanceSpotAdapter.__new__(BinanceSpotAdapter)
    adapter.rest = type("R", (), {})()

    with pytest.raises(NotImplementedError):
        await adapter.fetch_funding("BTC/USDT")
    with pytest.raises(NotImplementedError):
        await adapter.fetch_basis("BTC/USDT")
    with pytest.raises(NotImplementedError):
        await adapter.fetch_oi("BTC/USDT")


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
async def test_binance_futures_ws_stream_funding():
    ws = BinanceFuturesWSAdapter()

    msg = json.dumps({"data": {"r": "0.01", "i": "100", "E": 0}})

    async def _fake_messages(url, subscribe=None, ping_timeout=None):
        yield msg

    ws._ws_messages = _fake_messages
    gen = ws.stream_funding("BTC/USDT")
    funding = await gen.__anext__()
    await gen.aclose()
    assert funding["rate"] == 0.01
    assert funding["index_px"] == 100.0
    assert "interval_sec" not in funding


@pytest.mark.asyncio
async def test_binance_futures_ws_open_interest_timeout_recovery(monkeypatch, caplog):
    ws = BinanceFuturesWSAdapter()

    msg = json.dumps({"data": {"oi": "100", "E": 0, "s": "BTCUSDT"}})
    calls = {"per": 0, "arr": 0}

    async def _fake_messages(url, subscribe=None, ping_timeout=None):
        if "!openInterest@arr@" in url:
            calls["arr"] += 1
            yield msg
        else:
            calls["per"] += 1
            while True:
                await asyncio.sleep(0.02)

    ws._ws_messages = _fake_messages

    original_wait_for = asyncio.wait_for

    async def fake_wait_for(awaitable, timeout):
        return await original_wait_for(awaitable, 0.01)

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    caplog.set_level(logging.WARNING, logger="tradingbot.adapters.binance_futures_ws")

    gen = ws.stream_open_interest("BTC/USDT")
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.03)
    result = await task
    await gen.aclose()

    assert result["oi"] == 100.0
    assert any("No message received" in r.message for r in caplog.records)
    assert calls["arr"] == 1


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


@pytest.mark.asyncio
async def test_binance_spot_ws_stream_bba(monkeypatch):
    ws = BinanceSpotWSAdapter()

    msg = json.dumps({"data": {"b": "1", "B": "2", "a": "3", "A": "4"}})

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

    monkeypatch.setattr("websockets.connect", lambda url, **k: DummyWS([msg]))

    gen = ws.stream_bba("BTC/USDT")
    bba = await gen.__anext__()
    await gen.aclose()
    assert bba["bid_px"] == 1.0
    assert bba["bid_qty"] == 2.0
    assert bba["ask_px"] == 3.0
    assert bba["ask_qty"] == 4.0


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls", [OKXSpotAdapter, OKXFuturesAdapter])
async def test_okx_stream_bba(monkeypatch, adapter_cls):
    adapter = adapter_cls.__new__(adapter_cls)
    ExchangeAdapter.__init__(adapter)
    if adapter_cls is OKXFuturesAdapter:
        adapter.ws_public_url = "wss://example"
        msg = json.dumps({"data": [{"bids": [["1", "2"]], "asks": [["3", "4"]], "ts": "0"}]})

        async def fake_messages(url, sub, ping_timeout=None):
            yield msg

        adapter._ws_messages = fake_messages
    else:
        async def _ob(_symbol: str):
            yield {
                "bid_px": [1.0],
                "bid_qty": [2.0],
                "ask_px": [3.0],
                "ask_qty": [4.0],
                "ts": datetime.fromtimestamp(1, tz=timezone.utc),
            }

        monkeypatch.setattr(adapter, "stream_order_book", lambda s, depth=1: _ob(s))

    gen = adapter.stream_bba("BTC/USDT")
    bba = await gen.__anext__()
    await gen.aclose()
    assert bba["bid_px"] == 1.0
    assert bba["bid_qty"] == 2.0
    assert bba["ask_px"] == 3.0
    assert bba["ask_qty"] == 4.0


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_cls", [OKXSpotAdapter, OKXFuturesAdapter])
async def test_okx_stream_bba_handles_empty(monkeypatch, adapter_cls):
    adapter = adapter_cls.__new__(adapter_cls)
    ExchangeAdapter.__init__(adapter)
    if adapter_cls is OKXFuturesAdapter:
        adapter.ws_public_url = "wss://example"
        msg1 = json.dumps({"data": [{"bids": [], "asks": [], "ts": "0"}]})
        msg2 = json.dumps({"data": [{"bids": [["1", "2"]], "asks": [["3", "4"]], "ts": "1"}]})

        async def fake_messages(url, sub, ping_timeout=None):
            yield msg1
            yield msg2

        adapter._ws_messages = fake_messages

        gen = adapter.stream_bba("BTC/USDT")
        bba = await gen.__anext__()
        await gen.aclose()
        assert bba["bid_px"] == 1.0
        assert bba["bid_qty"] == 2.0
        assert bba["ask_px"] == 3.0
        assert bba["ask_qty"] == 4.0
    else:
        async def _ob(_symbol: str):
            yield {
                "bid_px": [],
                "bid_qty": [],
                "ask_px": [],
                "ask_qty": [],
                "ts": datetime.fromtimestamp(1, tz=timezone.utc),
            }

        monkeypatch.setattr(adapter, "stream_order_book", lambda s, depth=1: _ob(s))

        gen = adapter.stream_bba("BTC/USDT")
        bba = await gen.__anext__()
        await gen.aclose()
        assert bba["bid_px"] is None
        assert bba["bid_qty"] is None
        assert bba["ask_px"] is None
        assert bba["ask_qty"] is None


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


@pytest.mark.parametrize(
    "alias,instrument",
    [
        ("SOLUSDT", "SOL-PERPETUAL"),
        ("XRPUSDT", "XRP-PERPETUAL"),
        ("MATICUSDT", "MATIC-PERPETUAL"),
        ("DOTUSDT", "DOT-PERPETUAL"),
    ],
)
def test_deribit_symbol_map_new_instruments(alias, instrument):
    assert DeribitAdapter.normalize(alias) == instrument


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
