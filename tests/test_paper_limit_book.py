import asyncio

import pytest

from tradingbot.broker.broker import Broker
from tradingbot.execution.paper import PaperAdapter


@pytest.mark.asyncio
async def test_limit_order_partial_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1.0, price=90.0)
    assert res["status"] == "new"
    assert res["pending_qty"] == pytest.approx(1.0)

    fills = adapter.update_last_price("BTC/USDT", 89.0, qty=0.4)
    assert fills[0]["status"] == "partial"
    assert fills[0]["pending_qty"] == pytest.approx(0.6)

    fills = adapter.update_last_price("BTC/USDT", 88.0, qty=1.0)
    assert fills[0]["status"] == "filled"
    assert fills[0]["pending_qty"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_manual_requote_after_partial_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)

    # place initial order and receive partial fill
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1.0, price=90.0)
    assert res["status"] == "new"
    adapter.update_last_price("BTC/USDT", 89.0, qty=0.4)

    # re-quote remaining quantity at better price
    res2 = await adapter.place_order("BTC/USDT", "buy", "limit", 0.6, price=88.0)
    assert res2["status"] == "new"
    fills = adapter.update_last_price("BTC/USDT", 87.0, qty=1.0)
    assert fills[0]["status"] == "filled"
    assert fills[0]["qty"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_cancel_pending_order_reports_metrics():
    adapter = PaperAdapter(latency=0.01)
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1.0, price=90.0)
    await asyncio.sleep(0.02)
    cancel = await adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"
    assert cancel["time_in_book"] >= 0.02
    assert cancel["latency"] >= 0.01


@pytest.mark.asyncio
async def test_limit_fill_with_exact_cash_clears_reserve():
    symbol = "BTC/USDT"
    adapter = PaperAdapter()
    adapter.update_last_price(symbol, 100.0)
    maker_fee = adapter.maker_fee_bps / 10000.0
    requested_qty = 2.0
    accepted_qty = 1.0
    price = 90.0
    initial_cash = price * accepted_qty * (1 + maker_fee)
    adapter.state.cash = initial_cash
    adapter.account.cash = initial_cash

    broker = Broker(adapter)

    resp = await broker.place_limit(symbol, "buy", price, requested_qty)
    assert resp["status"] == "new"
    filled_qty = float(resp.get("filled_qty", 0.0))
    pending_qty = float(resp.get("pending_qty", 0.0))
    assert filled_qty == pytest.approx(0.0)
    assert pending_qty == pytest.approx(accepted_qty)

    account = adapter.account
    prev_pending = float(account.open_orders.get(symbol, {}).get("buy", 0.0))
    delta_open = pending_qty - prev_pending + filled_qty
    if abs(delta_open) > 1e-9:
        account.update_open_order(symbol, "buy", delta_open)

    # Order matches fully once price trades through the limit
    fills = adapter.update_last_price(symbol, price, qty=accepted_qty)
    assert fills and fills[0]["status"] == "filled"
    fill_qty = float(fills[0]["qty"])
    account.update_open_order(symbol, "buy", -fill_qty)

    assert account.open_orders == {}
    assert account.get_available_balance() == pytest.approx(account.cash)
    assert account.cash == pytest.approx(0.0)
