import asyncio
import time

import pytest

from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.order_types import Order
from tradingbot.config import settings
from tradingbot.utils.metrics import ROUTER_SELECTED_VENUE, ROUTER_STALE_BOOK


class FeeAdapter:
    def __init__(self):
        self.name = "fee"
        self.maker_fee_bps = 0.0
        self.taker_fee_bps = 0.0
        self.state = type("S", (), {"order_book": {}, "last_px": {}})()
        self.calls = 0

    async def place_order(self, **kwargs):  # pragma: no cover - not used
        return {"status": "filled", **kwargs}

    async def update_fees(self, symbol: str | None = None) -> None:
        self.calls += 1
        self.maker_fee_bps = 1.0
        self.taker_fee_bps = 2.0


class MockAdapter:
    def __init__(self, name, order_book):
        self.name = name
        self.state = type("S", (), {"order_book": order_book, "last_px": {}})()
        self.maker_fee_bps = 0.0
        self.taker_fee_bps = 0.0

    async def place_order(self, **kwargs):  # pragma: no cover - not used
        return {"status": "filled", **kwargs}

    async def update_fees(self, symbol: str | None = None) -> None:
        return None


@pytest.mark.asyncio
async def test_router_periodic_fee_update(monkeypatch):
    monkeypatch.setattr(settings, "router_fee_update_interval_sec", 0.1)
    adapter = FeeAdapter()
    router = ExecutionRouter(adapter)
    await asyncio.sleep(0.15)
    assert adapter.calls >= 1
    assert adapter.maker_fee_bps == 1.0
    assert adapter.taker_fee_bps == 2.0
    await router.close()


@pytest.mark.asyncio
async def test_best_venue_records_metrics(monkeypatch):
    ROUTER_SELECTED_VENUE.clear()
    ROUTER_STALE_BOOK.clear()
    ts = time.time()
    ob1 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)], "ts": ts - 10}}
    ob2 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)], "ts": ts}}
    a1 = MockAdapter("a1", ob1)
    a2 = MockAdapter("a2", ob2)
    monkeypatch.setattr(settings, "router_max_book_age_ms", 1000.0)
    router = ExecutionRouter([a1, a2], prefer="taker")
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    sel = await router.best_venue(order)
    assert sel is a2

    samples = list(ROUTER_STALE_BOOK.collect())[0].samples
    stale_sample = [s for s in samples if s.labels["venue"] == "a1"][0]
    assert stale_sample.value == 1.0

    samples = list(ROUTER_SELECTED_VENUE.collect())[0].samples
    selected_sample = [
        s
        for s in samples
        if s.labels["venue"] == "a2" and s.labels["path"] == "taker"
    ][0]
    assert selected_sample.value == 1.0
    await router.close()

