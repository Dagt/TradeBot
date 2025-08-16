import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.utils.metrics import SLIPPAGE


class MockAdapter:
    def __init__(self, name, order_book=None, fee_bps=0.0, latency=0.0, last_px=None, fill_price=None):
        self.name = name
        self.state = type("S", (), {"order_book": order_book or {}, "last_px": last_px or {}})
        self.fee_bps = fee_bps
        self.latency = latency
        self.fill_price = fill_price

    async def place_order(self, **kwargs):
        price = self.fill_price if self.fill_price is not None else kwargs.get("price")
        return {**kwargs, "status": "filled", "price": price}


@pytest.mark.asyncio
async def test_router_selects_lowest_cost_venue():
    ob1 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    ob2 = {"XYZ": {"bids": [(99.5, 1.0)], "asks": [(100.5, 1.0)]}}
    a1 = MockAdapter("a1", order_book=ob1, fee_bps=10.0, latency=5.0)
    a2 = MockAdapter("a2", order_book=ob2, fee_bps=0.0, latency=20.0)
    router = ExecutionRouter([a1, a2])
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    selected = await router.best_venue(order)
    assert selected is a2


@pytest.mark.asyncio
async def test_router_records_slippage_metric():
    SLIPPAGE.clear()
    ob = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    adapter = MockAdapter(
        "m",
        order_book=ob,
        last_px={"XYZ": 100.0},
        fill_price=101.0,
    )
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=1.0, price=100.0)
    await router.execute(order)
    samples = list(SLIPPAGE.collect())[0].samples
    count_sample = [
        s
        for s in samples
        if s.name == "order_slippage_bps_count" and s.labels["symbol"] == "XYZ" and s.labels["side"] == "buy"
    ][0]
    sum_sample = [
        s
        for s in samples
        if s.name == "order_slippage_bps_sum" and s.labels["symbol"] == "XYZ" and s.labels["side"] == "buy"
    ][0]
    assert count_sample.value == 1.0
    assert sum_sample.value == pytest.approx(100.0)
