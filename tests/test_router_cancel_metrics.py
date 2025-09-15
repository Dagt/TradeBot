import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.order_types import Order
from tradingbot.utils.metrics import CANCELS


class DummyAdapter:
    name = "dummy"

    def __init__(self):
        self._oid = 0

    async def place_order(self, **kwargs):  # pragma: no cover - simple stub
        self._oid += 1
        return {"status": "new", "order_id": str(self._oid)}

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        return {"status": "canceled", "order_id": order_id}


@pytest.mark.asyncio
async def test_cancel_increments_counter_once():
    CANCELS._value.set(0)
    adapter = DummyAdapter()
    router = ExecutionRouter(adapter)

    order1 = Order(symbol="BTC/USDT", side="buy", type_="limit", qty=1.0, price=100.0)
    res1 = await router.execute(order1)
    await router.cancel_order(res1["order_id"])
    assert CANCELS._value.get() == 1.0

    order2 = Order(symbol="BTC/USDT", side="buy", type_="limit", qty=1.0, price=100.0)
    res2 = await router.execute(order2)
    await router.cancel_order(res2["order_id"])
    assert CANCELS._value.get() == 2.0

