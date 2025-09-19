from datetime import timedelta
from types import MethodType

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
async def test_paper_order_clip_updates_pending_and_unlocks_cash():
    adapter = PaperAdapter(maker_fee_bps=250.0)
    adapter.state.cash = 100.0
    adapter.account.cash = 100.0
    symbol = "BTC/USDT"
    adapter.update_last_price(symbol, 100.0)

    guard = PortfolioGuard(GuardConfig(venue="paper"))
    risk = RiskService(guard, account=adapter.account)
    router = ExecutionRouter(adapter, risk_service=risk)

    requested_qty = 5.0
    order = Order(symbol=symbol, side="buy", type_="limit", qty=requested_qty, price=95.0)
    res = await router.execute(order)

    assert res["status"] == "new"
    clipped_qty = float(res["pending_qty"])
    expected_affordable = adapter.state.cash / (order.price * (1 + adapter.maker_fee_bps / 10000.0))
    assert clipped_qty == pytest.approx(expected_affordable)
    assert clipped_qty < requested_qty
    assert order.pending_qty == pytest.approx(clipped_qty)
    assert order.qty == pytest.approx(clipped_qty)

    locked_after_submit = risk.account.get_locked_usd(symbol)
    assert locked_after_submit == pytest.approx(clipped_qty * adapter.state.last_px[symbol])
    assert symbol in risk.account.open_orders
    assert risk.account.open_orders[symbol]["buy"] == pytest.approx(clipped_qty)

    events = adapter.update_last_price(symbol, 94.0, qty=clipped_qty)
    for ev in events:
        await router.handle_paper_event(ev)

    assert order.pending_qty == pytest.approx(0.0)
    assert symbol not in risk.account.open_orders
    assert risk.account.get_locked_usd(symbol) == pytest.approx(0.0)

    def _recalc_locked_total(account) -> float:
        total_locked = 0.0
        for sym, orders in getattr(account, "open_orders", {}).items():
            qty_total = 0.0
            if isinstance(orders, dict):
                for qty_val in orders.values():
                    try:
                        qty_total += abs(float(qty_val))
                    except (TypeError, ValueError):
                        continue
            else:
                try:
                    qty_total = abs(float(orders))
                except (TypeError, ValueError):
                    qty_total = 0.0
            price = float(account.prices.get(sym, 0.0))
            total_locked += qty_total * price
        return 0.0 if total_locked <= 1e-9 else total_locked

    locked_total = _recalc_locked_total(risk.account)
    assert locked_total == pytest.approx(0.0)


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


@pytest.mark.asyncio
async def test_router_forwards_timeout_and_logs_cancel_event(monkeypatch):
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    symbol = "BTC/USDT"
    adapter.update_last_price(symbol, 100.0)

    captured: dict[str, float | None] = {"timeout": None}
    original_place_order = adapter.place_order

    async def _tracking_place_order(self, *args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return await original_place_order(*args, **kwargs)

    monkeypatch.setattr(
        adapter,
        "place_order",
        MethodType(_tracking_place_order, adapter),
    )

    cancel_events: list[dict] = []

    def _on_expiry(order, res):
        cancel_events.append({"event": "cancel", "res": dict(res)})
        return None

    router = ExecutionRouter(adapter, on_order_expiry=_on_expiry)

    timeout = 0.05
    order = Order(
        symbol=symbol,
        side="buy",
        type_="limit",
        qty=1.0,
        price=90.0,
        time_in_force="GTD",
        timeout=timeout,
    )

    res = await router.execute(order)
    assert res["status"] == "new"
    assert captured["timeout"] == pytest.approx(timeout)

    order_id = res.get("order_id")
    book_for_symbol = adapter.state.book.get(symbol)
    assert book_for_symbol is not None and order_id in book_for_symbol
    book_order = book_for_symbol[order_id]
    book_order.placed_at = book_order.placed_at - timedelta(seconds=timeout + 0.05)

    events = adapter.update_last_price(symbol, adapter.state.last_px[symbol])
    assert events, "PaperAdapter should emit expiry events"
    assert cancel_events == []

    for event in events:
        assert event["status"] in {"expired", "partial"}
        await router.handle_paper_event(event)

    assert cancel_events and cancel_events[0]["event"] == "cancel"
