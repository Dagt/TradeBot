import json
import logging
import pytest
from datetime import datetime

from tradingbot.connectors import (
    DeribitConnector,
    Trade,
    OrderBook,
    Funding,
    Basis,
    OpenInterest,
)
from tradingbot.adapters import DeribitWSAdapter


class DummyRest:
    def __init__(self, trades, funding, basis, oi):
        self._trades = trades
        self._funding = funding
        self._basis = basis
        self._oi = oi

    async def fetch_trades(self, symbol):
        return self._trades

    async def fetch_funding_rate(self, symbol):
        return self._funding

    async def fetch_basis(self, symbol):
        return self._basis

    async def fetch_open_interest(self, symbol):
        return self._oi


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

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_rest_normalization_deribit():
    trades = [{"timestamp": 1000, "price": "1", "amount": "2", "direction": "buy"}]
    funding = {"fundingRate": "0.01", "timestamp": 1000}
    basis = {"basis": "5", "timestamp": 1000}
    oi = {"openInterest": "200", "timestamp": 1000}
    c = DeribitConnector()
    c.rest = DummyRest(trades, funding, basis, oi)

    res_trades = await c.fetch_trades("BTC-PERPETUAL")
    assert isinstance(res_trades[0], Trade)
    assert res_trades[0].price == 1.0

    res_funding = await c.fetch_funding("BTC-PERPETUAL")
    assert isinstance(res_funding, Funding)
    assert res_funding.rate == 0.01

    res_basis = await c.fetch_basis("BTC-PERPETUAL")
    assert isinstance(res_basis, Basis)
    assert res_basis.basis == 5.0

    res_oi = await c.fetch_open_interest("BTC-PERPETUAL")
    assert isinstance(res_oi, OpenInterest)
    assert res_oi.oi == 200.0


@pytest.mark.asyncio
async def test_stream_trades_and_orderbook(monkeypatch):
    c = DeribitConnector()
    trade_msgs = [
        json.dumps({"params": {"data": [{"price": "1", "amount": "1", "direction": "buy", "timestamp": 0}]}}),
        json.dumps({"params": {"data": [{"price": "3", "amount": "1", "direction": "sell", "timestamp": 1}]}}),
    ]
    ob_msgs = [
        json.dumps({"params": {"data": {"bids": [["1", "1"]], "asks": [["2", "2"]], "timestamp": 0}}})
    ]
    ws_iter = iter([DummyWS(trade_msgs.copy()), DummyWS(ob_msgs.copy())])

    def fake_connect(url):
        return next(ws_iter)

    monkeypatch.setattr("websockets.connect", fake_connect)

    tgen = c.stream_trades("BTC-PERPETUAL")
    t1 = await tgen.__anext__()
    t2 = await tgen.__anext__()
    assert t1.price == 1.0
    assert t2.side == "sell"
    await tgen.aclose()

    ogen = c.stream_order_book("BTC-PERPETUAL")
    ob = await ogen.__anext__()
    assert isinstance(ob, OrderBook)
    assert ob.bids[0][0] == 1.0
    await ogen.aclose()


@pytest.mark.asyncio
async def test_deribit_ws_adapter_parsing(monkeypatch, caplog):
    adapter = DeribitWSAdapter(rest=DummyRest([], {}, {}, {}))

    trade_msgs = [
        json.dumps({"error": {"message": "bad"}}),
        json.dumps(
            {
                "params": {
                    "channel": "trades.BTC-PERPETUAL.raw",
                    "data": [
                        {
                            "price": "100",
                            "amount": "0.5",
                            "direction": "buy",
                            "timestamp": 0,
                        }
                    ],
                }
            }
        ),
    ]

    book_msgs = [
        json.dumps({"error": {"message": "bad"}}),
        json.dumps(
            {
                "params": {
                    "channel": "book.BTC-PERPETUAL.none.10.100ms",
                    "data": {
                        "bids": [["100", "1"]],
                        "asks": [["101", "2"]],
                        "timestamp": 0,
                    },
                }
            }
        ),
    ]

    ws_iter = iter([DummyWS(trade_msgs.copy()), DummyWS(book_msgs.copy())])

    def fake_connect(*args, **kwargs):
        return next(ws_iter)

    monkeypatch.setattr("websockets.connect", fake_connect)

    with caplog.at_level(logging.ERROR):
        tgen = adapter.stream_trades("BTC-PERPETUAL")
        trade = await tgen.__anext__()
        assert trade["price"] == 100.0
        assert trade["qty"] == 0.5
        await tgen.aclose()

        ogen = adapter.stream_order_book("BTC-PERPETUAL", depth=10)
        ob = await ogen.__anext__()
        assert ob["bid_px"][0] == 100.0
        assert ob["ask_qty"][0] == 2.0
        await ogen.aclose()

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) >= 2


@pytest.mark.asyncio
async def test_deribit_ws_adapter_channel_mismatch():
    adapter = DeribitWSAdapter()

    async def fake_messages(url, sub):
        yield json.dumps({"params": {"channel": "other", "data": []}})

    adapter._ws_messages = fake_messages

    gen = adapter.stream_trades("BTC-PERPETUAL")
    with pytest.raises(ValueError):
        await gen.__anext__()


@pytest.mark.asyncio
async def test_deribit_ws_adapter_delegates_to_rest():
    class _Rest:
        async def stream_order_book(self, symbol, depth=10):
            yield {"bid_px": [1.0], "ask_px": [2.0], "bid_qty": [3.0], "ask_qty": [4.0]}

        async def fetch_funding(self, symbol):
            return {"rate": 0.01}

        async def fetch_basis(self, symbol):
            return {"basis": 5.0}

        async def fetch_oi(self, symbol):
            return {"oi": 10.0}

    rest = _Rest()
    adapter = DeribitWSAdapter(rest=rest)

    funding = await adapter.fetch_funding("BTC-PERPETUAL")
    assert funding["rate"] == 0.01

    basis = await adapter.fetch_basis("BTC-PERPETUAL")
    assert basis["basis"] == 5.0

    oi = await adapter.fetch_oi("BTC-PERPETUAL")
    assert oi["oi"] == 10.0

    gen = adapter.stream_order_book("BTC-PERPETUAL")
    ob = await gen.__anext__()
    await gen.aclose()
    assert ob["bid_px"][0] == 1.0
