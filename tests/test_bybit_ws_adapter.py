import pytest
from datetime import datetime

from tradingbot.adapters.bybit_ws import BybitWSAdapter


class DummyRest:
    def __init__(self):
        self.created = None
        self.canceled = None

    def fetchFundingRate(self, symbol):
        assert symbol == "BTCUSDT"
        return {"fundingRate": "0.01", "timestamp": 1000}

    def publicGetV5MarketOpenInterest(self, params):
        assert params == {"category": "linear", "symbol": "BTCUSDT"}
        return {"result": {"list": [{"openInterest": "123.45", "timestamp": 1000}]}}

    def place_order(self, symbol, side, type_, qty, price=None, params=None):
        self.created = (symbol, side, type_, qty, price, params or {})
        return {"id": "1"}

    def cancel_order(self, order_id, symbol=None):
        self.canceled = (order_id, symbol)
        return {"status": "canceled"}


@pytest.mark.asyncio
async def test_fetch_funding_oi_and_orders():
    rest = DummyRest()
    adapter = BybitWSAdapter(rest=rest)

    async def _req(fn, *a, **k):
        return fn(*a, **k)

    adapter._request = _req  # bypass throttling

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
