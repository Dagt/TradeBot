import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.storage import timescale
from tradingbot.live.common_exec import persist_after_order


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
async def test_persist_after_order_records_reduce_only(monkeypatch):
    captured = {}

    def fake_insert_fill(engine, **kwargs):
        nonlocal captured
        captured = kwargs

    monkeypatch.setattr("tradingbot.live.common_exec.insert_order", lambda *a, **k: None)
    monkeypatch.setattr("tradingbot.live.common_exec.insert_fill", fake_insert_fill)
    monkeypatch.setattr("tradingbot.live.common_exec.upsert_position", lambda *a, **k: None)
    monkeypatch.setattr("tradingbot.live.common_exec.insert_pnl_snapshot", lambda *a, **k: None)
    monkeypatch.setattr("tradingbot.live.common_exec.insert_risk_event", lambda *a, **k: None)

    pos_obj = type("P", (), {"qty": 0.0, "avg_price": 0.0, "realized_pnl": 0.0, "fees_paid": 0.0})()

    persist_after_order(
        "eng",
        venue="v",
        strategy="s",
        symbol="X",
        side="buy",
        type_="market",
        qty=1.0,
        mark_price=100.0,
        resp={"status": "filled"},
        reduce_only=True,
        pos_obj=pos_obj,
    )

    assert captured["reduce_only"] is True
