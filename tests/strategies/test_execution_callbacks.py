import pytest

from tradingbot.strategies.base import Strategy, Signal, record_signal_metrics
from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter


class DummyStrategy(Strategy):
    name = "dummy"

    @record_signal_metrics
    def on_bar(self, bar):
        # Always emit the side provided in the bar for testing
        return Signal(bar.get("side", "buy"), 1.0)


def setup_strategy(side="buy"):
    strat = DummyStrategy()
    # Record last signal so callbacks can evaluate the edge
    strat.on_bar({"symbol": "XYZ", "exchange": "ex", "close": 100.0, "side": side})
    return strat


class PartialAdapter:
    def __init__(self):
        self.calls = []
        self.name = "p"

    async def place_order(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            filled = kwargs["qty"] / 2
            return {"status": "partial", "filled_qty": filled, "price": kwargs.get("price", 100.0)}
        return {"status": "filled", "price": kwargs.get("price", 100.0)}


class ExpiringAdapter:
    def __init__(self):
        self.calls = []
        self.name = "e"

    async def place_order(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"status": "expired", "price": kwargs.get("price", 100.0)}
        return {"status": "filled", "price": kwargs.get("price", 100.0)}


@pytest.mark.asyncio
async def test_requote_on_partial_fill_when_edge_persists():
    strat = setup_strategy("buy")
    adapter = PartialAdapter()
    router = ExecutionRouter(adapter, on_partial_fill=strat.on_partial_fill)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=10.0, price=100.0)
    res = await router.execute(order)
    assert res["status"] == "filled"
    # pending_qty retains remaining amount after the partial fill
    assert strat.pending_qty["XYZ"] == pytest.approx(5.0)
    assert len(adapter.calls) == 2
    assert adapter.calls[1]["qty"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_cancel_on_partial_fill_when_edge_gone():
    strat = setup_strategy("sell")  # last signal opposite to order
    adapter = PartialAdapter()
    router = ExecutionRouter(adapter, on_partial_fill=strat.on_partial_fill)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=10.0, price=100.0)
    res = await router.execute(order)
    assert res["status"] == "partial"
    assert strat.pending_qty["XYZ"] == pytest.approx(5.0)
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_order_expiry_requote_when_edge_persists():
    strat = setup_strategy("buy")
    strat._last_atr = {"XYZ": 1.0}
    adapter = ExpiringAdapter()
    router = ExecutionRouter(adapter, on_order_expiry=strat.on_order_expiry)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=10.0, price=100.0)
    res = await router.execute(order)
    assert res["status"] == "filled"
    # pending_qty mirrors the size of the re-quoted order
    assert strat.pending_qty["XYZ"] == pytest.approx(10.0)
    assert len(adapter.calls) == 2
    # re-quoted price includes offset
    assert adapter.calls[1]["price"] == pytest.approx(100.1)


@pytest.mark.asyncio
async def test_order_expiry_cancel_when_edge_gone():
    strat = setup_strategy("sell")
    adapter = ExpiringAdapter()
    router = ExecutionRouter(adapter, on_order_expiry=strat.on_order_expiry)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=10.0, price=100.0)
    res = await router.execute(order)
    assert res["status"] == "expired"
    assert strat.pending_qty["XYZ"] == pytest.approx(10.0)
    assert len(adapter.calls) == 1


class FallbackStrategy(DummyStrategy):
    def on_partial_fill(self, order: Order, res: dict):
        action = super().on_partial_fill(order, res)
        return action or "taker"


@pytest.mark.asyncio
async def test_partial_fill_no_fallback_when_edge_gone():
    strat = FallbackStrategy()
    # record opposite last signal to indicate edge vanished
    strat.on_bar({"symbol": "XYZ", "exchange": "ex", "close": 100.0, "side": "sell"})
    adapter = PartialAdapter()
    router = ExecutionRouter(adapter, on_partial_fill=strat.on_partial_fill)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=10.0, price=100.0)
    res = await router.execute(order)
    assert res["status"] == "partial"
    assert len(adapter.calls) == 1
