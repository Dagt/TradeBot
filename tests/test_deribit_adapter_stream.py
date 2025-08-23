import pytest
from tradingbot.adapters import DeribitAdapter
from tradingbot.adapters.base import ExchangeAdapter


class DummyRest:
    def __init__(self):
        self.calls = 0

    async def fetch_trades(self, symbol, limit=1):
        self.calls += 1
        if self.calls == 1:
            return [{"id": "1", "price": "10", "amount": "0.5", "timestamp": 1000, "direction": "buy"}]
        elif self.calls == 2:
            return [{"id": "1", "price": "10", "amount": "0.5", "timestamp": 1000, "direction": "buy"}]
        else:
            return [{"id": "2", "price": "11", "amount": "0.4", "timestamp": 2000, "direction": "sell"}]


@pytest.mark.asyncio
async def test_stream_trades_dedup(monkeypatch):
    adapter = DeribitAdapter.__new__(DeribitAdapter)
    ExchangeAdapter.__init__(adapter, rate_limit_per_sec=1e6)
    adapter.rest = DummyRest()

    async def no_sleep(_):
        pass

    monkeypatch.setattr("tradingbot.adapters.deribit.asyncio.sleep", no_sleep)
    monkeypatch.setattr("tradingbot.adapters.base.asyncio.sleep", no_sleep)

    gen = adapter.stream_trades("BTC-PERPETUAL")
    t1 = await gen.__anext__()
    t2 = await gen.__anext__()
    await gen.aclose()

    assert t1["price"] == 10.0
    assert t2["price"] == 11.0
    assert t2["side"] == "sell"
