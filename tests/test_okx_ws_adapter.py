import json
import pytest
from datetime import datetime

from tradingbot.adapters.okx_ws import OKXWSAdapter


class DummyRest:
    def __init__(self):
        self.created = None
        self.canceled = None

    def fetchFundingRate(self, symbol):
        assert symbol == "BTCUSDT"
        return {"fundingRate": "0.01", "timestamp": 1000}

    def publicGetPublicOpenInterest(self, params):
        assert params == {"instId": "BTCUSDT"}
        return {"data": [{"oi": "123.45", "ts": 1000}]}

    def place_order(self, symbol, side, type_, qty, price=None, params=None):
        self.created = (symbol, side, type_, qty, price, params or {})
        return {"id": "1"}

    def cancel_order(self, order_id, symbol=None):
        self.canceled = (order_id, symbol)
        return {"status": "canceled"}


@pytest.mark.parametrize(
    "symbol, suffix",
    [
        ("BTC-PERP", "PERP"),
        ("BTC-FOO", "FOO"),
    ],
)
def test_normalize_symbol_invalid(symbol, suffix):
    adapter = OKXWSAdapter()
    with pytest.raises(ValueError) as excinfo:
        adapter.normalize_symbol(symbol)
    assert suffix in str(excinfo.value)


@pytest.mark.asyncio
async def test_fetch_funding_oi_and_orders():
    rest = DummyRest()
    adapter = OKXWSAdapter(rest=rest)

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req

    funding = await adapter.fetch_funding("BTC/USDT")
    assert funding["rate"] == 0.01
    assert isinstance(funding["ts"], datetime)

    oi = await adapter.fetch_oi("BTC/USDT")
    assert oi["oi"] == 123.45
    assert isinstance(oi["ts"], datetime)

    order = await adapter.place_order("BTC/USDT", "buy", "market", 1)
    assert rest.created[0] == "BTC/USDT"
    assert order["id"] == "1"

    cancel = await adapter.cancel_order("1", "BTC/USDT")
    assert rest.canceled == ("1", "BTC/USDT")
    assert cancel["status"] == "canceled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,args,channel,msg",
    [
        (
            "stream_trades",
            (),
            "trades",
            json.dumps({"data": [{"px": "1", "sz": "1", "ts": "0", "side": "buy"}]}),
        ),
        (
            "stream_order_book",
            (5,),
            "books5",
            json.dumps({"data": [{"bids": [["1", "1"]], "asks": [["2", "2"]], "ts": "0"}]}),
        ),
        (
            "stream_bba",
            (),
            "books5",
            json.dumps({"data": [{"bids": [["1", "2"]], "asks": [["3", "4"]], "ts": "0"}]}),
        ),
        (
            "stream_funding",
            (),
            "funding-rate",
            json.dumps(
                {
                    "data": [
                        {"fundingRate": "0.01", "fundingInterval": "60", "ts": "0"}
                    ]
                }
            ),
        ),
        (
            "stream_open_interest",
            (),
            "open-interest",
            json.dumps({"data": [{"oi": "5", "ts": "0"}]}),
        ),
    ],
)
async def test_subscription_format(method, args, channel, msg):
    adapter = OKXWSAdapter()
    captured: dict[str, dict] = {}

    async def fake_messages(url, sub, ping_timeout=None):
        captured["sub"] = json.loads(sub)
        yield msg

    adapter._ws_messages = fake_messages

    gen = getattr(adapter, method)("BTC/USDT", *args)
    await gen.__anext__()
    await gen.aclose()

    expected_sym = adapter._normalize("BTC/USDT")
    assert captured["sub"] == {
        "op": "subscribe",
        "args": [{"channel": channel, "instId": expected_sym}],
    }


@pytest.mark.asyncio
async def test_stream_bba_discard_invalid(caplog):
    adapter = OKXWSAdapter()
    events = [
        json.dumps({"data": [{"bids": [["0", "1"]], "asks": [["2", "2"]], "ts": "0"}]}),
        json.dumps({"data": [{"bids": [["1", "1"]], "asks": [], "ts": "0"}]}),
        json.dumps({"data": [{"bids": [["1", "2"]], "asks": [["3", "4"]], "ts": "0"}]})
    ]

    async def fake_messages(url, sub, ping_timeout=None):
        for e in events:
            yield e

    adapter._ws_messages = fake_messages
    caplog.set_level("WARNING")
    gen = adapter.stream_bba("BTC/USDT")
    result = await gen.__anext__()
    await gen.aclose()

    assert result["bid_px"] == pytest.approx(1.0)
    discarded = [r for r in caplog.records if "Discarding BBA event" in r.message]
    assert len(discarded) == 2
