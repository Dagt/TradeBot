import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.storage import timescale
from tradingbot.execution.paper import PaperAdapter
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


class DummyAdapter:
    def __init__(self, name="a", maker_fee_bps=0.0, taker_fee_bps=0.0):
        self.name = name
        self.state = type("S", (), {"order_book": {}, "last_px": {}})()
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps

    async def place_order(self, **kwargs):
        return {"status": "filled", **kwargs, "price": kwargs.get("price")}


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


@pytest.mark.asyncio
async def test_execute_persists_fee_type(monkeypatch):
    captured = {}

    def fake_insert_order(engine, **kwargs):
        nonlocal captured
        captured = kwargs

    monkeypatch.setattr(timescale, "insert_order", fake_insert_order)
    adapter = DummyAdapter(taker_fee_bps=10.0)
    router = ExecutionRouter(adapter, storage_engine="eng")
    order = Order(symbol="X", side="buy", type_="market", qty=1.0)
    await router.execute(order)
    assert captured["notes"]["fee_type"] == "taker"
    assert captured["notes"]["fee_bps"] == 10.0


@pytest.mark.asyncio
async def test_execute_persists_maker_fee(monkeypatch):
    captured = {}

    def fake_insert_order(engine, **kwargs):
        nonlocal captured
        captured = kwargs

    monkeypatch.setattr(timescale, "insert_order", fake_insert_order)
    adapter = DummyAdapter(maker_fee_bps=1.5)
    router = ExecutionRouter(adapter, storage_engine="eng")
    order = Order(
        symbol="X",
        side="buy",
        type_="limit",
        qty=1.0,
        price=1.0,
        post_only=True,
    )
    await router.execute(order)
    assert captured["notes"]["fee_type"] == "maker"
    assert captured["notes"]["fee_bps"] == 1.5


@pytest.mark.asyncio
async def test_execute_persists_reduce_only(monkeypatch):
    captured = {}

    def fake_insert_order(engine, **kwargs):
        nonlocal captured
        captured = kwargs

    monkeypatch.setattr(timescale, "insert_order", fake_insert_order)
    adapter = DummyAdapter()
    router = ExecutionRouter(adapter, storage_engine="eng")
    order = Order(symbol="X", side="buy", type_="market", qty=1.0, reduce_only=True)
    await router.execute(order)
    assert captured["notes"]["reduce_only"] is True


@pytest.mark.asyncio
async def test_router_syncs_guard_account_on_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    symbol = "BTCUSDT"
    adapter.update_last_price(symbol, 100.0)
    guard = PortfolioGuard(GuardConfig(venue="paper"))
    risk = RiskService(guard, account=adapter.account)
    router = ExecutionRouter(adapter, risk_service=risk)
    order_buy = Order(symbol=symbol, side="buy", type_="limit", qty=1.0, price=100.0)
    res_buy = await router.execute(order_buy)
    assert res_buy["status"] == "filled"
    assert risk.account.positions[symbol] == pytest.approx(1.0)
    assert guard.st.venue_positions["paper"][symbol] == pytest.approx(1.0)
    order_sell = Order(symbol=symbol, side="sell", type_="limit", qty=1.0, price=100.0)
    res_sell = await router.execute(order_sell)
    assert res_sell["status"] == "filled"
    assert risk.account.positions.get(symbol, 0.0) == pytest.approx(0.0)
    assert guard.st.venue_positions["paper"].get(symbol, 0.0) == pytest.approx(0.0)
