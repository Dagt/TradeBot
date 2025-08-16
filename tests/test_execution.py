import pytest
from hypothesis import given, strategies as st

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter


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


@pytest.mark.asyncio
async def test_twap_splits_orders(paper_adapter):
    paper_adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper_adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=4.0)
    res = await router.execute(order, algo="twap", slices=4)
    assert len(res) == 4
    assert paper_adapter.state.pos["BTCUSDT"].qty == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_vwap_distribution(paper_adapter):
    paper_adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper_adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=6.0)
    res = await router.execute(order, algo="vwap", volumes=[1, 2, 3])
    assert len(res) == 3
    assert paper_adapter.state.pos["BTCUSDT"].qty == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_pov_participates(paper_adapter):
    paper_adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper_adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=5.0)

    async def trades():
        for q in [4.0, 4.0, 4.0]:
            yield {"ts": 0, "price": 100.0, "qty": q, "side": "buy"}

    res = await router.execute(
        order,
        algo="pov",
        participation_rate=0.5,
        trades=trades(),
    )
    assert paper_adapter.state.pos["BTCUSDT"].qty == pytest.approx(5.0)
    assert len(res) >= 2


class DummyExecAdapter:
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
