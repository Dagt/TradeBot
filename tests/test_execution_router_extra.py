import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter


class DummyAdapter:
    def __init__(self, name="a"):
        self.name = name
        self.state = type("S", (), {"order_book": {}, "last_px": {}})()

    async def place_order(self, **kwargs):
        return {"status": "ok", **kwargs}


@pytest.mark.asyncio
async def test_best_venue_fallback():
    a = DummyAdapter("a")
    b = DummyAdapter("b")
    router = ExecutionRouter([a, b])
    order = Order(symbol="X", side="buy", type_="market", qty=1.0)
    selected = await router.best_venue(order)
    assert selected is a


@pytest.mark.asyncio
async def test_unknown_algo_raises():
    adapter = DummyAdapter()
    router = ExecutionRouter(adapter)
    order = Order(symbol="X", side="buy", type_="market", qty=1.0)
    with pytest.raises(ValueError):
        await router.execute(order, algo="foo")
