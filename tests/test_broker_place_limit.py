import pytest

from tradingbot.execution.paper import PaperAdapter
from tradingbot.broker import Broker


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
async def test_place_limit_tif_expiry():
    adapter = PaperAdapter()
    adapter.state.cash = 1000.0
    adapter.update_last_price("BTC/USDT", 100.0)
    broker = Broker(adapter)
    res = await broker.place_limit("BTC/USDT", "buy", 90.0, 1.0, tif="GTD:0.01|PO")
    assert res["filled_qty"] == 0.0
    assert res["time_in_book"] >= 0.01
