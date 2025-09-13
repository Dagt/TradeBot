import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.broker.broker import Broker
from tradingbot.utils.metrics import ORDERS, SKIPS


class DummyAdapter:
    name = "dummy"
    maker_fee_bps = 0.0

    def __init__(self, status="filled"):
        self.status = status

    async def place_order(self, *args, **kwargs):
        return {
            "status": self.status,
            "order_id": "1",
            "filled_qty": kwargs.get("qty", 0.0),
            "price": kwargs.get("price"),
        }


@pytest.mark.asyncio
async def test_orders_counter_incremented():
    ORDERS._value.set(0)
    adapter = DummyAdapter()
    router = ExecutionRouter(adapter)
    broker = Broker(router)
    await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert ORDERS._value.get() == 1.0


@pytest.mark.asyncio
async def test_skips_counter_incremented():
    ORDERS._value.set(0)
    SKIPS._value.set(0)
    adapter = DummyAdapter(status="rejected")
    router = ExecutionRouter(adapter)
    broker = Broker(router)
    await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert ORDERS._value.get() == 0.0
    assert SKIPS._value.get() == 1.0
