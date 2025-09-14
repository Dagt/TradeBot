import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.order_types import Order
from tradingbot.execution.normalize import SymbolRules


class DummyMeta:
    def __init__(self, rules):
        self.rules = rules
        self.client = type("C", (), {"symbols": list(rules.keys())})()

    def rules_for(self, symbol):
        return self.rules[symbol]


class DummyAdapter:
    def __init__(self, rules):
        self.meta = DummyMeta(rules)
        self.state = type("S", (), {"order_book": {}, "last_px": {}})()
        self.name = "d"
        self.maker_fee_bps = 0.0
        self.taker_fee_bps = 0.0

    async def place_order(self, **kwargs):
        return {"status": "filled", **kwargs}


@pytest.mark.asyncio
async def test_router_uses_tick_size_per_symbol():
    rules = {
        "AAA/USDT": SymbolRules(price_step=0.5),
        "BBB/USDT": SymbolRules(price_step=0.1),
    }
    adapter = DummyAdapter(rules)
    router = ExecutionRouter(adapter)

    order_a = Order(symbol="AAA/USDT", side="buy", type_="limit", qty=1.0, price=1.23)
    res_a = await router.execute(order_a)
    assert res_a["price"] == pytest.approx(1.0)

    order_b = Order(symbol="BBB/USDT", side="buy", type_="limit", qty=1.0, price=1.23)
    res_b = await router.execute(order_b)
    assert res_b["price"] == pytest.approx(1.2)
