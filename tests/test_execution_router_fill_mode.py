import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.storage import timescale


class MockAdapter:
    def __init__(self, order_book=None, last_bar=None):
        self.name = "m"
        self.state = type("S", (), {
            "order_book": order_book or {},
            "last_bar": last_bar or {},
            "last_px": {},
        })()

    async def place_order(self, **kwargs):
        return {"status": "filled", "price": kwargs.get("price", 100.0)}


@pytest.mark.asyncio
async def test_fill_price_bidask_with_slip():
    ob = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    adapter = MockAdapter(order_book=ob)
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    res = await router.execute(order, fill_mode="bidask", slip_bps=50.0)
    expected = 101.0 * 1.005
    assert res["price"] == pytest.approx(expected)
    assert res["fill_price"] == res["price"]




@pytest.mark.asyncio
async def test_fill_price_persisted(monkeypatch):
    ob = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    adapter = MockAdapter(order_book=ob)
    router = ExecutionRouter(adapter)
    recorded = {}

    def fake_insert_order(engine, *, strategy, exchange, symbol, side, type_, qty, px, status, ext_order_id=None, notes=None):
        recorded["px"] = px

    monkeypatch.setattr(timescale, "insert_order", fake_insert_order)
    router._engine = object()
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    res = await router.execute(order, fill_mode="bidask")
    assert recorded["px"] == res["price"]
