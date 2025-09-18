import asyncio

import pytest

from tradingbot.broker import Broker
from tradingbot.execution.paper import PaperAdapter
from tradingbot.strategies.base import Strategy, Signal, record_signal_metrics
from tradingbot.filters.liquidity import LiquidityFilterManager
from tradingbot.utils.metrics import FILL_COUNT, CANCELS


class DummyAdapter:
    """Adapter que nunca llena órdenes y contabiliza intentos."""

    def __init__(self):
        self.place_order_calls = 0
        self.maker_fee_bps = 0.0

    async def place_order(self, *args, **kwargs):
        self.place_order_calls += 1
        return {"order_id": self.place_order_calls, "status": "new", "filled": 0.0}

    async def cancel_order(self, *args, **kwargs):
        return {"status": "canceled"}


@pytest.mark.asyncio
async def test_place_limit_immediate_fill():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    broker = Broker(adapter)
    res = await broker.place_limit("BTC/USDT", "buy", 100.0, 1.0)
    assert res["status"] == "filled"
    assert res["filled_qty"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_place_limit_replaces_on_expiry():
    adapter = DummyAdapter()
    broker = Broker(adapter)
    await broker.place_limit("BTC/USDT", "buy", 90.0, 1.0, tif="GTD:0.01|PO")
    assert adapter.place_order_calls == 5


def test_broker_maker_fee_override():
    adapter = PaperAdapter(maker_fee_bps=2.0)
    broker = Broker(adapter, maker_fee_bps=1.0)
    assert broker.maker_fee_bps == 1.0


class PartialAdapter:
    def __init__(self):
        self.calls = []
        self.maker_fee_bps = 0.0

    async def place_order(self, *args, **kwargs):
        qty = args[3] if len(args) > 3 else kwargs.get("qty", 0.0)
        self.calls.append({"qty": qty, **kwargs})
        if len(self.calls) == 1:
            return {"status": "partial", "qty": qty / 2, "order_id": 1}
        return {"status": "filled", "qty": qty, "order_id": 2}

    async def cancel_order(self, *args, **kwargs):
        return {"status": "canceled"}


class ExpiringAdapter:
    def __init__(self):
        self.calls = []
        self.maker_fee_bps = 0.0
        self.cancel_calls = 0

    async def place_order(self, *args, **kwargs):
        qty = args[3] if len(args) > 3 else kwargs.get("qty", 0.0)
        price = args[4] if len(args) > 4 else kwargs.get("price")
        data = {"qty": qty, "price": price}
        data.update(kwargs)
        self.calls.append(data)
        if len(self.calls) == 1:
            return {"status": "new", "qty": 0.0, "order_id": 1}
        return {"status": "filled", "qty": qty, "order_id": 2}

    async def cancel_order(self, *args, **kwargs):
        self.cancel_calls += 1
        return {"status": "canceled"}


liquidity = LiquidityFilterManager()


class DummyStrategy(Strategy):
    name = "dummy"

    @record_signal_metrics(liquidity)
    def on_bar(self, bar):
        return Signal(bar.get("side", "buy"), 1.0)


def setup_strategy(side="buy"):
    strat = DummyStrategy()
    strat.on_bar({"symbol": "BTC/USDT", "exchange": "ex", "close": 100.0, "side": side})
    return strat


@pytest.mark.asyncio
async def test_place_limit_partial_fill_requotes():
    strat = setup_strategy("buy")
    adapter = PartialAdapter()
    broker = Broker(adapter)
    res = await broker.place_limit(
        "BTC/USDT",
        "buy",
        100.0,
        10.0,
        on_partial_fill=strat.on_partial_fill,
    )
    assert res["status"] == "filled"
    assert strat.pending_qty["BTC/USDT"] == pytest.approx(5.0)
    assert len(adapter.calls) == 2
    assert adapter.calls[1]["qty"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_place_limit_partial_fill_no_fallback():
    adapter = PartialAdapter()
    broker = Broker(adapter)

    def ignore(order, res):
        return "taker"  # any non re_quote value

    res = await broker.place_limit(
        "BTC/USDT",
        "buy",
        100.0,
        10.0,
        on_partial_fill=ignore,
    )
    assert res["status"] == "partial"
    assert res["pending_qty"] == pytest.approx(5.0)
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_place_limit_expiry_respects_callback():
    strat = setup_strategy("buy")
    strat._last_atr = {"BTC/USDT": 1.0}
    adapter = ExpiringAdapter()
    broker = Broker(adapter)
    await broker.place_limit(
        "BTC/USDT",
        "buy",
        90.0,
        1.0,
        tif="GTD:0.01",
        on_order_expiry=strat.on_order_expiry,
    )
    assert len(adapter.calls) == 2
    assert strat.pending_qty["BTC/USDT"] == pytest.approx(1.0)
    assert adapter.calls[1]["price"] == pytest.approx(90.1)


@pytest.mark.asyncio
async def test_place_limit_expiry_cancels_on_edge_gone():
    strat = setup_strategy("sell")
    adapter = ExpiringAdapter()
    broker = Broker(adapter)
    await broker.place_limit(
        "BTC/USDT",
        "buy",
        90.0,
        1.0,
        tif="GTD:0.01",
        on_order_expiry=strat.on_order_expiry,
    )
    assert len(adapter.calls) == 1
    assert adapter.cancel_calls == 1
    assert strat.pending_qty["BTC/USDT"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_place_limit_gtd_timeout_increments_cancel_metric():
    CANCELS._value.set(0)
    adapter = ExpiringAdapter()
    broker = Broker(adapter)

    def stop_on_expiry(order, res):
        return None

    await broker.place_limit(
        "BTC/USDT",
        "buy",
        90.0,
        1.0,
        tif="GTD:0.01",
        on_order_expiry=stop_on_expiry,
    )

    assert adapter.cancel_calls == 1
    assert CANCELS._value.get() == 1.0


@pytest.mark.asyncio
async def test_place_limit_paper_fill_before_gtd():
    symbol = "BTC/USDT"
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price(symbol, 100.0)
    broker = Broker(adapter)

    FILL_COUNT.clear()

    task = asyncio.create_task(
        broker.place_limit(symbol, "buy", 99.0, 1.0, tif="GTD:0.2|PO")
    )

    await asyncio.sleep(0)
    events = broker.update_last_price(symbol, 98.5)
    assert events and events[0]["status"] == "filled"

    res = await task
    assert res["status"] == "filled"
    assert res["filled_qty"] == pytest.approx(1.0)
    assert res["pending_qty"] == pytest.approx(0.0)

    samples = list(FILL_COUNT.collect())[0].samples
    fill_sample = [
        s
        for s in samples
        if s.name == "order_fills_total"
        and s.labels.get("symbol") == symbol
        and s.labels.get("side") == "buy"
    ][0]
    assert fill_sample.value == 1.0
