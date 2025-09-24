import asyncio

import pytest

from tradingbot.execution.paper import PaperAdapter, PaperPosition, PRICE_TOLERANCE
from tradingbot.backtesting.engine import SlippageModel


@pytest.mark.asyncio
async def test_post_only_limit_rests_then_maker_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)

    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 1.0, price=105.0, post_only=True
    )
    assert res["status"] == "new"
    assert res["pending_qty"] == pytest.approx(1.0)

    fill_price = res["price"]
    fills = adapter.update_last_price("BTC/USDT", fill_price, qty=1.0)
    fill = [f for f in fills if f.get("order_id") == res["order_id"]][0]
    assert fill["status"] == "filled"
    assert fill["qty"] == pytest.approx(1.0)
    assert fill["fee_type"] == "maker"


@pytest.mark.asyncio
async def test_post_only_limit_adjusts_price_to_book_best_ask():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    book = {"bid": 99.0, "ask": 101.0}
    res = await adapter.place_order(
        "BTC/USDT",
        "buy",
        "limit",
        1.0,
        price=105.0,
        post_only=True,
        book=book,
    )
    assert res["status"] == "new"
    assert res["price"] == pytest.approx(book["ask"] - PRICE_TOLERANCE)
    assert res["pending_qty"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_post_only_limit_rejects_when_best_ask_non_positive():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 1.0)
    book = {"bid": 0.0, "ask": 0.0}
    res = await adapter.place_order(
        "BTC/USDT",
        "buy",
        "limit",
        1.0,
        price=0.01,
        post_only=True,
        book=book,
    )
    assert res["status"] == "rejected"
    assert res["reason"] == "post_only_invalid_reference"


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
async def test_partial_cancel_reports_filled_and_pending():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1.0, price=90.0)
    adapter.update_last_price("BTC/USDT", 89.0, qty=0.4)
    cancel = await adapter.cancel_order(res["order_id"], "BTC/USDT")
    assert cancel["status"] == "partial"
    assert cancel["filled_qty"] == pytest.approx(0.4)
    assert cancel["pending_qty"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_market_slippage_matches_model():
    model = SlippageModel(volume_impact=0.1, pct=0.0)
    adapter = PaperAdapter(slippage_model=model)
    adapter.state.cash = 10000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    book = {"bid": 99.5, "ask": 100.5, "bid_size": 10.0, "ask_size": 10.0, "volume": 1000.0}
    qty = 2.0
    res = await adapter.place_order("BTC/USDT", "buy", "market", qty, book=book)
    assert res["status"] == "filled"
    expected_price = model.adjust("buy", qty, 100.0, book)
    assert res["price"] == pytest.approx(expected_price)
    expected_slip = (expected_price - 100.0) / 100.0 * 10000
    assert res["slippage_bps"] == pytest.approx(expected_slip)


@pytest.mark.asyncio
async def test_limit_slippage_matches_model():
    model = SlippageModel(volume_impact=0.1, pct=0.0)
    adapter = PaperAdapter(slippage_model=model)
    adapter.state.cash = 10000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 5.0, price=99.0)
    assert res["status"] == "new"
    book = {"bid": 97.5, "ask": 98.5, "bid_size": 10.0, "ask_size": 3.0, "volume": 1000.0}
    fills = adapter.update_last_price("BTC/USDT", 98.0, book=book)
    fill = fills[0]
    assert fill["status"] == "partial"
    exp_px, exp_qty, _ = model.fill("buy", 5.0, 98.0, book)
    assert fill["price"] == pytest.approx(exp_px)
    assert fill["qty"] == pytest.approx(exp_qty)
    assert fill["pending_qty"] == pytest.approx(5.0 - exp_qty)
    exp_slip = (exp_px - 98.0) / 98.0 * 10000
    assert fill["slippage_bps"] == pytest.approx(exp_slip)


@pytest.mark.asyncio
async def test_timeout_emits_partial_cancel():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order("BTC/USDT", "buy", "limit", 1.0, price=90.0, timeout=0.01)
    adapter.update_last_price("BTC/USDT", 89.0, qty=0.4)
    await asyncio.sleep(0.02)
    events = adapter.update_last_price("BTC/USDT", 100.0)
    cancel = [e for e in events if e.get("order_id") == res["order_id"]][0]
    assert cancel["status"] == "partial"
    assert cancel["pending_qty"] == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_limit_order_queue_simulation():
    model = SlippageModel(volume_impact=0.0, pct=0.0)
    adapter = PaperAdapter(slippage_model=model)
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 5.0, price=99.0, book={"ask_size": 5.0}
    )
    assert res["status"] == "new"
    fills_partial = adapter.update_last_price(
        "BTC/USDT", 99.0, qty=5.0, book={"ask_size": 8.0, "volume": 100.0}
    )
    partial = fills_partial[0]
    assert partial["status"] == "partial"
    assert partial["qty"] == pytest.approx(3.0)
    assert partial["pending_qty"] == pytest.approx(2.0)
    fills_final = adapter.update_last_price(
        "BTC/USDT", 99.0, qty=2.0, book={"ask_size": 2.0, "volume": 100.0}
    )
    final = fills_final[0]
    assert final["status"] == "filled"
    assert final["qty"] == pytest.approx(2.0)
    assert final["pending_qty"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_timeout_emits_expired():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    res = await adapter.place_order(
        "BTC/USDT", "buy", "limit", 1.0, price=90.0, timeout=0.01
    )
    await asyncio.sleep(0.02)
    events = adapter.update_last_price("BTC/USDT", 100.0)
    cancel = [e for e in events if e.get("order_id") == res["order_id"]][0]
    assert cancel["status"] == "expired"
    assert cancel["filled_qty"] == pytest.approx(0.0)
    assert cancel["pending_qty"] == pytest.approx(1.0)


def test_realized_pnl_on_close():
    adapter = PaperAdapter()
    # cerrar largo
    adapter.state.pos["BTC/USDT"] = PaperPosition(qty=1.0, avg_px=100.0)
    adapter._apply_fill("BTC/USDT", "sell", 1.0, 110.0, False)
    assert adapter.state.realized_pnl == pytest.approx(10.0)
    assert adapter.state.pos["BTC/USDT"].qty == pytest.approx(0.0)
    assert adapter.state.pos["BTC/USDT"].avg_px == pytest.approx(0.0)

    # cerrar corto
    adapter.state.pos["ETH/USDT"] = PaperPosition(qty=-1.0, avg_px=100.0)
    prev = adapter.state.realized_pnl
    adapter._apply_fill("ETH/USDT", "buy", 1.0, 90.0, False)
    assert adapter.state.realized_pnl - prev == pytest.approx(10.0)
    assert adapter.state.pos["ETH/USDT"].qty == pytest.approx(0.0)
    assert adapter.state.pos["ETH/USDT"].avg_px == pytest.approx(0.0)
