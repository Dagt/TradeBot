import pytest
from datetime import datetime, timezone

from tradingbot.adapters.binance import BinanceWSAdapter


class DummyRest:
    def __init__(self):
        self.created = None
        self.canceled = None

    # Funding rate endpoint
    def fetchFundingRate(self, symbol):
        assert symbol == "BTCUSDT"
        return {"fundingRate": "0.01", "timestamp": 1000}

    # Open interest endpoint
    def fapiPublicGetOpenInterest(self, params):
        assert params == {"symbol": "BTCUSDT"}
        return {"openInterest": "123.45", "time": 1000}

    # Order placement
    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        self.created = (symbol, type_, side, amount, price, params or {})
        return {"id": "1"}

    # Order cancellation
    def cancel_order(self, order_id, symbol=None):
        self.canceled = (order_id, symbol)
        return {"status": "canceled"}


@pytest.mark.asyncio
async def test_fetch_funding_and_oi_and_orders():
    rest = DummyRest()
    adapter = BinanceWSAdapter(rest=rest)

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
