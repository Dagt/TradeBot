import json
import asyncio

import pytest

from tradingbot.connectors import (
    BinanceConnector,
    BybitConnector,
    OKXConnector,
    OrderBook,
    Trade,
    Funding,
    OpenInterest,
)


class DummyRest:
    def __init__(self, trades, funding, oi):
        self._trades = trades
        self._funding = funding
        self._oi = oi

    async def fetch_trades(self, symbol):
        return self._trades

    async def fetch_funding_rate(self, symbol):
        return self._funding

    async def fetch_open_interest(self, symbol):
        return self._oi


class DummyWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []
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


@pytest.mark.parametrize(
    "connector_cls",
    [BinanceConnector, BybitConnector, OKXConnector],
)
@pytest.mark.asyncio
async def test_rest_normalization(connector_cls):
    trades = [{"timestamp": 1000, "price": "1", "amount": "2", "side": "buy"}]
    funding = {"fundingRate": "0.01", "timestamp": 1000}
    oi = {"openInterest": "5", "timestamp": 1000}
    c = connector_cls()
    c.rest = DummyRest(trades, funding, oi)

    res_trades = await c.fetch_trades("BTC/USDT")
    assert isinstance(res_trades[0], Trade)
    assert res_trades[0].price == 1.0

    res_funding = await c.fetch_funding("BTC/USDT")
    assert isinstance(res_funding, Funding)
    assert res_funding.rate == 0.01

    res_oi = await c.fetch_open_interest("BTC/USDT")
    assert isinstance(res_oi, OpenInterest)
    assert res_oi.oi == 5.0


@pytest.mark.asyncio
async def test_stream_reconnect(monkeypatch, caplog):
    caplog.set_level("WARNING")
    c = BinanceConnector()
    c.ping_interval = 0
    msg1 = json.dumps({"b": [["1", "1"]], "a": [["2", "2"]]})
    msg2 = json.dumps({"b": [["3", "3"]], "a": [["4", "4"]]})
    ws1 = DummyWS([msg1])
    ws2 = DummyWS([msg2])
    ws_iter = iter([ws1, ws2])

    def fake_connect(url, *_, **__):
        return next(ws_iter)

    monkeypatch.setattr("websockets.connect", fake_connect)

    sleeps: list[float] = []

    async def fake_sleep(t):
        sleeps.append(t)

    monkeypatch.setattr("tradingbot.connectors.base.asyncio.sleep", fake_sleep)

    gen = c.stream_order_book("BTCUSDT")
    book1 = await gen.__anext__()
    book2 = await gen.__anext__()
    assert isinstance(book1, OrderBook)
    assert book1.bids[0][0] == 1.0
    assert book2.bids[0][0] == 3.0
    await gen.aclose()

    assert any("ws_reconnect" in r.message for r in caplog.records)
    assert len(sleeps) == 1
    assert 0.5 <= sleeps[0] <= 1.5


@pytest.mark.asyncio
async def test_trade_stream_reconnect(monkeypatch, caplog):
    caplog.set_level("WARNING")
    c = BinanceConnector()
    c.ping_interval = 0
    msg1 = json.dumps({"p": "1", "q": "1", "T": 0, "m": False})
    msg2 = json.dumps({"p": "3", "q": "1", "T": 1, "m": True})
    ws1 = DummyWS([msg1])
    ws2 = DummyWS([msg2])
    ws_iter = iter([ws1, ws2])

    def fake_connect(url, *_, **__):
        return next(ws_iter)

    monkeypatch.setattr("websockets.connect", fake_connect)

    async def no_ping(ws):
        pass

    monkeypatch.setattr(c, "_ping", no_ping)

    gen = c.stream_trades("BTCUSDT")
    trade1 = await gen.__anext__()
    trade2 = await gen.__anext__()
    assert isinstance(trade1, Trade)
    assert trade1.price == 1.0
    assert trade2.price == 3.0
    await gen.aclose()

    assert any("ws_reconnect" in r.message for r in caplog.records)


binance_msgs = [
    json.dumps({"p": "1", "q": "1", "T": 0, "m": False}),
    json.dumps({"p": "3", "q": "1", "T": 1, "m": True}),
]

bybit_msgs = [
    json.dumps({"data": [{"p": "1", "v": "1", "t": 0, "S": "Buy"}]}),
    json.dumps({"data": [{"p": "3", "v": "1", "t": 1, "S": "Sell"}]}),
]

okx_msgs = [
    json.dumps({"data": [{"px": "1", "sz": "1", "ts": "0", "side": "buy"}]}),
    json.dumps({"data": [{"px": "3", "sz": "1", "ts": "1", "side": "sell"}]}),
]


@pytest.mark.parametrize(
    "connector_cls,messages",
    [
        (BinanceConnector, binance_msgs),
        (BybitConnector, bybit_msgs),
        (OKXConnector, okx_msgs),
    ],
)
@pytest.mark.asyncio
async def test_stream_trades(connector_cls, messages, monkeypatch):
    c = connector_cls()
    ws = DummyWS(messages.copy())

    def fake_connect(url, *_, **__):
        return ws

    monkeypatch.setattr("websockets.connect", fake_connect)

    gen = c.stream_trades("BTCUSDT")
    trade1 = await gen.__anext__()
    trade2 = await gen.__anext__()
    assert isinstance(trade1, Trade)
    assert trade1.price == 1.0
    assert trade2.price == 3.0
    await gen.aclose()
    if connector_cls is OKXConnector:
        assert json.loads(ws.sent[0]) == {
            "op": "subscribe",
            "args": [{"channel": "trades", "instId": "BTCUSDT"}],
        }
    else:
        assert ws.sent[0] == c._ws_trades_subscribe("BTCUSDT")


class DummyOrderRest:
    """Minimal REST client mocking order and balance endpoints."""

    def __init__(self) -> None:
        self.last_create = None
        self.last_cancel = None

    async def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.last_create = (symbol, type_, side, amount, price, params)
        return {
            "id": "1",
            "status": "open",
            "symbol": symbol,
            "side": side,
            "price": price,
            "amount": amount,
        }

    async def cancel_order(self, order_id, symbol=None):
        self.last_cancel = (order_id, symbol)
        return {"id": order_id, "status": "canceled"}

    async def fetch_balance(self):
        return {"BTC": {"total": "1"}, "USDT": {"total": "2"}}


@pytest.mark.parametrize("connector_cls", [BybitConnector, OKXConnector, BinanceConnector])
@pytest.mark.asyncio
async def test_order_and_balance_parsing(connector_cls):
    c = connector_cls()
    rest = DummyOrderRest()
    c.rest = rest

    res = await c.place_order(
        "BTC/USDT",
        "buy",
        "limit",
        1,
        price=10,
        post_only=True,
        time_in_force="FOK",
        iceberg_qty=0.5,
    )
    assert res["id"] == "1"
    assert res["price"] == 10.0
    assert rest.last_create[5].get("postOnly") is True or rest.last_create[5].get("timeInForce") == "GTX"
    assert rest.last_create[5]["timeInForce"] in {"FOK", "GTX"}
    assert rest.last_create[5].get("icebergQty", rest.last_create[5].get("iceberg")) == 0.5

    cancel = await c.cancel_order("1", "BTC/USDT")
    assert cancel["status"] == "canceled"
    assert rest.last_cancel == ("1", "BTC/USDT")

    bal = await c.fetch_balance()
    assert bal == {"BTC": 1.0, "USDT": 2.0}
