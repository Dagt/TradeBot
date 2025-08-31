import pytest

from tradingbot.broker import Broker
from tradingbot.execution.paper import PaperAdapter
from tradingbot.strategies.base import Strategy, Signal, record_signal_metrics


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
    assert res["time_in_book"] >= 0.0


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
        self.calls.append({"qty": qty, **kwargs})
        if len(self.calls) == 1:
            return {"status": "new", "qty": 0.0, "order_id": 1}
        return {"status": "filled", "qty": qty, "order_id": 2}

    async def cancel_order(self, *args, **kwargs):
        self.cancel_calls += 1
        return {"status": "canceled"}


class DummyStrategy(Strategy):
    name = "dummy"

    @record_signal_metrics
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
async def test_place_limit_expiry_respects_callback():
    strat = setup_strategy("buy")
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
