import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.paper import PaperAdapter
from tradingbot.execution.router import ExecutionRouter
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService


@pytest.mark.asyncio
async def test_paper_partial_fill_remainder_cancelled():
    adapter = PaperAdapter(min_notional=50.0)
    adapter.state.cash = 1000.0
    symbol = "BTC/USDT"
    adapter.update_last_price(symbol, 100.0)
    guard = PortfolioGuard(GuardConfig(venue="paper"))
    risk = RiskService(guard, account=adapter.account)
    router = ExecutionRouter(adapter, risk_service=risk)

    order = Order(symbol=symbol, side="buy", type_="limit", qty=1.0, price=90.0)
    res = await router.execute(order)
    assert res["status"] == "new"
    assert risk.account.open_orders[symbol]["buy"] == pytest.approx(order.qty)

    events = adapter.update_last_price(symbol, 89.0, qty=0.6)
    assert events

    handled_results = []
    for ev in events:
        res_event = await router.handle_paper_event(ev)
        if res_event is not None:
            handled_results.append(res_event)

    assert handled_results, "Expected router to process paper events"
    final = handled_results[-1]
    assert final["pending_qty"] == pytest.approx(0.0)
    assert final.get("remaining_cancelled") is True
    assert risk.account.get_locked_usd(symbol) == pytest.approx(0.0)
    assert symbol not in risk.account.open_orders
