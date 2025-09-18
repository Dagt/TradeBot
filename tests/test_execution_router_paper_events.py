import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.paper import PaperAdapter
from tradingbot.execution.router import ExecutionRouter
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.base import Strategy


class StubStrategy(Strategy):
    name = "stub"

    def on_bar(self, bar):
        return None

    def __init__(self):
        self.partial_called = False
        self.expiry_called = False

    def on_partial_fill(self, order, res):
        self.partial_called = True
        return None

    def on_order_expiry(self, order, res):
        self.expiry_called = True
        return None


@pytest.mark.asyncio
async def test_partial_fill_event_triggers_callback():
    strat = StubStrategy()
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    router = ExecutionRouter(adapter, on_partial_fill=strat.on_partial_fill)

    order = Order(symbol="BTC/USDT", side="buy", type_="limit", qty=1.0, price=90.0)
    res = await router.execute(order)
    assert res["status"] == "new"

    events = adapter.update_last_price("BTC/USDT", 89.0, qty=0.4)
    for ev in events:
        await router.handle_paper_event(ev)
    assert strat.partial_called is True


@pytest.mark.asyncio
async def test_open_order_locked_notional_tracks_fill_lifecycle():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    symbol = "BTC/USDT"
    adapter.update_last_price(symbol, 100.0)
    guard = PortfolioGuard(GuardConfig(venue="paper"))
    risk = RiskService(guard, account=adapter.account)
    router = ExecutionRouter(adapter, risk_service=risk)

    order = Order(symbol=symbol, side="buy", type_="limit", qty=1.0, price=90.0)
    res = await router.execute(order)
    assert res["status"] == "new"

    locked_after_submit = risk.account.get_locked_usd(symbol)
    assert locked_after_submit == pytest.approx(100.0)
    assert symbol in risk.account.open_orders
    assert risk.account.open_orders[symbol]["buy"] == pytest.approx(order.qty)

    events = adapter.update_last_price(symbol, 89.0, qty=order.qty)
    for ev in events:
        await router.handle_paper_event(ev)

    assert risk.account.get_locked_usd(symbol) == pytest.approx(0.0)
    assert symbol not in risk.account.open_orders


@pytest.mark.asyncio
async def test_expiry_event_triggers_callback():
    strat = StubStrategy()
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    router = ExecutionRouter(adapter, on_order_expiry=strat.on_order_expiry)

    order = Order(symbol="BTC/USDT", side="buy", type_="limit", qty=1.0, price=90.0)
    order.timeout = 0.0
    res = await router.execute(order)
    assert res["status"] == "new"

    events = adapter.update_last_price("BTC/USDT", 100.0)
    for ev in events:
        await router.handle_paper_event(ev)
    assert strat.expiry_called is True
