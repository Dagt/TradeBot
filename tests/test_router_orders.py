import logging
import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter


class MockAdapter:
    def __init__(self, name, order_book, fee_bps=0.0, latency=0.0):
        self.name = name
        self.state = type("S", (), {"order_book": order_book})
        self.fee_bps = fee_bps
        self.latency = latency
        self.kwargs = None

    async def place_order(self, **kwargs):
        self.kwargs = kwargs
        book = self.state.order_book[kwargs["symbol"]]
        price = kwargs.get("price")
        if price is None:
            price = book["asks"][0][0] if kwargs["side"] == "buy" else book["bids"][0][0]
        return {"status": "filled", "price": price, **kwargs}


@pytest.mark.asyncio
async def test_best_venue_selection():
    ob = {
        "XYZ": {"bids": [(99.5, 1.0)], "asks": [(100.5, 1.0)]},
    }
    ob2 = {
        "XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]},
    }
    fast = MockAdapter("fast", ob, fee_bps=10.0, latency=20.0)
    slow = MockAdapter("slow", ob2, fee_bps=0.0, latency=10.0)
    router = ExecutionRouter([fast, slow])
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    selected = await router.best_venue(order)
    assert selected is fast


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"type_": "limit", "price": 100.0},
        {"type_": "limit", "price": 100.0, "post_only": True},
        {"type_": "limit", "price": 100.0, "time_in_force": "IOC"},
        {"type_": "limit", "price": 100.0, "time_in_force": "FOK"},
        {"type_": "limit", "price": 100.0, "iceberg_qty": 0.1},
        {"type_": "market", "reduce_only": True},
    ],
)
async def test_order_type_support(kwargs):
    ob = {
        "XYZ": {"bids": [(99.0, 5.0)], "asks": [(101.0, 5.0), (102.0, 5.0)]}
    }
    adapter = MockAdapter("m", ob)
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", qty=1.0, **kwargs)
    res = await router.execute(order)
    for key, val in kwargs.items():
        assert adapter.kwargs[key] == val
    assert "queue_position" in res
    assert "est_slippage_bps" in res


class FailingAdapter:
    name = "failing"

    async def place_order(self, **_):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_router_logs_order_error(caplog):
    adapter = FailingAdapter()
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    with caplog.at_level(logging.ERROR):
        res = await router.execute(order)
    assert res["status"] == "error"
    assert "order placement failed" in caplog.text.lower()
