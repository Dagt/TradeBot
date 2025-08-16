import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.algos import TWAP, VWAP, POV


@pytest.mark.asyncio
async def test_paper_adapter_execution(paper_adapter, mock_order):
    paper_adapter.update_last_price(mock_order["symbol"], 100.0)

    res = await paper_adapter.place_order(**mock_order)
    assert res["status"] == "filled"
    assert paper_adapter.state.pos[mock_order["symbol"]].qty == mock_order["qty"]

    cancel = await paper_adapter.cancel_order(res["order_id"])
    assert cancel["status"] == "canceled"


@pytest.mark.asyncio
async def test_twap_splits_orders(paper_adapter):
    paper_adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper_adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=4.0)
    algo = TWAP(router, slices=4)
    res = await algo.execute(order)
    assert len(res) == 4
    assert paper_adapter.state.pos["BTCUSDT"].qty == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_vwap_distribution(paper_adapter):
    paper_adapter.update_last_price("BTCUSDT", 100.0)
    router = ExecutionRouter(paper_adapter)
    order = Order(symbol="BTCUSDT", side="buy", type_="market", qty=6.0)
    algo = VWAP(router, volumes=[1, 2, 3])
    res = await algo.execute(order)
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

    algo = POV(router, participation_rate=0.5)
    res = await algo.execute(order, trades())
    assert paper_adapter.state.pos["BTCUSDT"].qty == pytest.approx(5.0)
    assert len(res) >= 2
