import json

import pytest

from tradingbot.connectors import (
    BinanceConnector,
    BybitConnector,
    OKXConnector,
    OrderBook,
    Trade,
    Funding,
)


class DummyRest:
    def __init__(self, trades, funding):
        self._trades = trades
        self._funding = funding

    async def fetch_trades(self, symbol):
        return self._trades

    async def fetch_funding_rate(self, symbol):
        return self._funding


class DummyWS:
    def __init__(self, messages):
        self.messages = messages
        self.sent = []

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


@pytest.mark.parametrize(
    "connector_cls",
    [BinanceConnector, BybitConnector, OKXConnector],
)
@pytest.mark.asyncio
async def test_rest_normalization(connector_cls):
    trades = [{"timestamp": 1000, "price": "1", "amount": "2", "side": "buy"}]
    funding = {"fundingRate": "0.01", "timestamp": 1000}
    c = connector_cls()
    c.rest = DummyRest(trades, funding)

    res_trades = await c.fetch_trades("BTC/USDT")
    assert isinstance(res_trades[0], Trade)
    assert res_trades[0].price == 1.0

    res_funding = await c.fetch_funding("BTC/USDT")
    assert isinstance(res_funding, Funding)
    assert res_funding.rate == 0.01


@pytest.mark.asyncio
async def test_stream_reconnect(monkeypatch, caplog):
    caplog.set_level("WARNING")
    c = BinanceConnector()
    msg1 = json.dumps({"b": [["1", "1"]], "a": [["2", "2"]]})
    msg2 = json.dumps({"b": [["3", "3"]], "a": [["4", "4"]]})
    ws_iter = iter([DummyWS([msg1]), DummyWS([msg2])])

    def fake_connect(url):
        return next(ws_iter)

    monkeypatch.setattr("websockets.connect", fake_connect)

    gen = c.stream_order_book("BTCUSDT")
    book1 = await gen.__anext__()
    book2 = await gen.__anext__()
    assert isinstance(book1, OrderBook)
    assert book1.bids[0][0] == 1.0
    assert book2.bids[0][0] == 3.0
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

    def fake_connect(url):
        return ws

    monkeypatch.setattr("websockets.connect", fake_connect)

    gen = c.stream_trades("BTCUSDT")
    trade1 = await gen.__anext__()
    trade2 = await gen.__anext__()
    assert isinstance(trade1, Trade)
    assert trade1.price == 1.0
    assert trade2.price == 3.0
    await gen.aclose()
    assert ws.sent[0] == c._ws_trades_subscribe("BTCUSDT")
