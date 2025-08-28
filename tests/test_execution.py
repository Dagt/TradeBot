import time

import pytest
from hypothesis import given, strategies as st

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.paper import PaperAdapter


class TimedPaperAdapter(PaperAdapter):
    def __init__(self):
        super().__init__()
        self.state.cash = 1_000_000.0
        self.call_times: list[float] = []

    async def place_order(self, *args, **kwargs):  # type: ignore[override]
        self.call_times.append(time.monotonic())
        return await super().place_order(*args, **kwargs)


@pytest.mark.asyncio
async def test_paper_adapter_execution(paper_adapter, mock_order):
    paper_adapter.update_last_price(mock_order["symbol"], 100.0)

    res = await paper_adapter.place_order(**mock_order)
    assert res["status"] == "filled"
    assert paper_adapter.state.pos[mock_order["symbol"]].qty == mock_order["qty"]

    cancel = await paper_adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"


@pytest.mark.asyncio
async def test_paper_stream_updates_price(paper_adapter):
    trades = [
        {"ts": 1, "price": 100.0, "qty": 1.0, "side": "buy"},
        {"ts": 2, "price": 101.0, "qty": 1.0, "side": "sell"},
    ]

    gen = paper_adapter.stream_trades("BTCUSDT", trades)
    first = await gen.__anext__()

    assert first["price"] == 100.0
    assert paper_adapter.state.last_px["BTCUSDT"] == 100.0

    res = await paper_adapter.place_order("BTCUSDT", "buy", "market", 0.5)
    assert res["status"] == "filled"
    assert res["price"] == 100.0


def test_paper_adapter_uses_config_fees():
    from tradingbot.config import settings

    adapter = PaperAdapter()
    assert adapter.maker_fee_bps == settings.paper_maker_fee_bps
    assert adapter.taker_fee_bps == settings.paper_taker_fee_bps


@pytest.mark.asyncio
async def test_twap_splits_orders():
    adapter = TimedPaperAdapter()
    adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=4.0)
    res = await router.execute(order, algo="twap", slices=4, delay=0.01)
    assert len(res) == 4
    assert [r["qty"] for r in res] == [pytest.approx(1.0)] * 4
    assert adapter.state.pos["BTCUSDT"].qty == pytest.approx(4.0)
    diffs = [t2 - t1 for t1, t2 in zip(adapter.call_times, adapter.call_times[1:])]
    assert all(d >= 0.01 for d in diffs)


@pytest.mark.asyncio
async def test_vwap_distribution():
    adapter = TimedPaperAdapter()
    adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=6.0)
    res = await router.execute(order, algo="vwap", volumes=[1, 2, 3], delay=0.01)
    assert len(res) == 3
    assert [r["qty"] for r in res] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]
    assert adapter.state.pos["BTCUSDT"].qty == pytest.approx(6.0)
    diffs = [t2 - t1 for t1, t2 in zip(adapter.call_times, adapter.call_times[1:])]
    assert all(d >= 0.01 for d in diffs)


@pytest.mark.asyncio
async def test_pov_participates():
    adapter = TimedPaperAdapter()
    adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=5.0)

    trades = [{"ts": 0, "price": 100.0, "qty": q, "side": "buy"} for q in [4.0, 4.0, 4.0]]

    res = await router.execute(
        order,
        algo="pov",
        participation_rate=0.5,
        trades=trades,
    )
    assert adapter.state.pos["BTCUSDT"].qty == pytest.approx(5.0)
    assert [r["qty"] for r in res] == [pytest.approx(2.0), pytest.approx(2.0), pytest.approx(1.0)]
    diffs = [t2 - t1 for t1, t2 in zip(adapter.call_times, adapter.call_times[1:])]
    assert all(d < 0.05 for d in diffs)


class DummyExecAdapter:
    name = "dummy"

    def __init__(self):
        self.args = None

    async def place_order(self, **kwargs):
        self.args = kwargs
        return kwargs


@given(
    st.builds(
        Order,
        symbol=st.sampled_from(["BTCUSDT", "ETHUSDT"]),
        side=st.sampled_from(["buy", "sell"]),
        type_=st.sampled_from(["market", "limit"]),
        qty=st.floats(min_value=0.1, max_value=5.0),
        price=st.one_of(st.none(), st.floats(min_value=10, max_value=1000)),
        post_only=st.booleans(),
        time_in_force=st.one_of(st.none(), st.sampled_from(["GTC", "IOC"])),
    )
)
@pytest.mark.asyncio
async def test_execution_router_property(order):
    adapter = DummyExecAdapter()
    router = ExecutionRouter(adapter)
    res = await router.execute(order)
    assert adapter.args["symbol"] == order.symbol
    assert res["side"] == order.side
