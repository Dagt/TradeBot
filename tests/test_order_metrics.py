import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.broker.broker import Broker
from tradingbot.utils.metrics import ORDER_SENT, ORDER_REJECTS


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
async def test_order_sent_counter_incremented():
    ORDER_SENT._value.set(0)
    adapter = DummyAdapter()
    router = ExecutionRouter(adapter)
    broker = Broker(router)
    await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert ORDER_SENT._value.get() == 1.0


@pytest.mark.asyncio
async def test_order_rejects_counter_incremented():
    ORDER_SENT._value.set(0)
    ORDER_REJECTS._value.set(0)
    adapter = DummyAdapter(status="rejected")
    router = ExecutionRouter(adapter)
    broker = Broker(router)
    await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert ORDER_SENT._value.get() == 1.0
    assert ORDER_REJECTS._value.get() == 1.0
