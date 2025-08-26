import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.order_types import Order
from tradingbot.risk.manager import RiskManager


class DummyAdapter:
    name = "dummy"

    async def place_order(self, **kwargs):
        return {"status": "filled", **kwargs}


@pytest.mark.asyncio
async def test_router_closes_position_on_stop_loss():
    adapter = DummyAdapter()
    rm = RiskManager(risk_pct=0.05)
    rm.set_position(1.0)
    rm.check_limits(100.0)
    router = ExecutionRouter(adapter, risk_manager=rm)
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0, price=90.0)
    res = await router.execute(order)
    assert res["status"] == "stop_loss"
    assert res["exit"]["reduce_only"] is True
    assert res["exit"]["side"] == "sell"
    assert rm.enabled
