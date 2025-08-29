import pytest

from tradingbot.execution.paper import PaperAdapter
from tradingbot.broker import Broker


class DummyAdapter:
    """Adapter que nunca llena órdenes y contabiliza intentos."""

    def __init__(self):
        self.place_order_calls = 0
        self.maker_fee_bps = 0.0

    async def place_order(self, *args, **kwargs):
        self.place_order_calls += 1
        return {"order_id": self.place_order_calls, "status": "new", "filled": 0.0}

    async def cancel_order(self, *args, **kwargs):
        return {"status": "canceled"}


@pytest.mark.asyncio
async def test_place_limit_immediate_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    broker = Broker(adapter)
    res = await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert res["status"] == "filled"
    assert res["filled_qty"] == pytest.approx(1.0)
    assert res["time_in_book"] >= 0.0


@pytest.mark.asyncio
async def test_place_limit_replaces_on_expiry():
    adapter = DummyAdapter()
    broker = Broker(adapter)
    await broker.place_limit("BTC/USDT", "buy", 90.0, 1.0, tif="GTD:0.01|PO")
    assert adapter.place_order_calls == 5


def test_broker_maker_fee_override():
    adapter = PaperAdapter(maker_fee_bps=2.0)
    broker = Broker(adapter, maker_fee_bps=1.0)
    assert broker.maker_fee_bps == 1.0
