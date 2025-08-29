import pytest

from tradingbot.execution.order_types import Order
from tradingbot.execution.router import ExecutionRouter
from tradingbot.execution.slippage import impact_by_depth, queue_position
from tradingbot.utils.metrics import SLIPPAGE
from tradingbot.backtesting.engine import SlippageModel
from tradingbot.data.features import calc_ofi
import pandas as pd


class MockAdapter:
    def __init__(
        self,
        name,
        order_book=None,
        maker_fee_bps: float = 0.0,
        taker_fee_bps: float = 0.0,
        latency: float = 0.0,
        last_px=None,
        fill_price=None,
    ):
        self.name = name
        self.state = type("S", (), {"order_book": order_book or {}, "last_px": last_px or {}})
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.latency = latency
        self.fill_price = fill_price

    async def place_order(self, **kwargs):
        price = self.fill_price if self.fill_price is not None else kwargs.get("price")
        return {**kwargs, "status": "filled", "price": price}


@pytest.mark.asyncio
async def test_router_selects_lowest_cost_venue():
    ob1 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    ob2 = {"XYZ": {"bids": [(99.5, 1.0)], "asks": [(100.5, 1.0)]}}
    a1 = MockAdapter("a1", order_book=ob1, taker_fee_bps=10.0, latency=5.0)
    a2 = MockAdapter("a2", order_book=ob2, taker_fee_bps=0.0, latency=20.0)
    router = ExecutionRouter([a1, a2], prefer="taker")
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    selected = await router.best_venue(order)
    assert selected is a2


@pytest.mark.asyncio
async def test_router_selects_lowest_cost_venue_maker():
    ob1 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    ob2 = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(101.0, 1.0)]}}
    a1 = MockAdapter("a1", order_book=ob1, maker_fee_bps=0.0)
    a2 = MockAdapter("a2", order_book=ob2, maker_fee_bps=5.0)
    router = ExecutionRouter([a1, a2], prefer="maker")
    order = Order(
        symbol="XYZ",
        side="buy",
        type_="limit",
        qty=1.0,
        price=100.0,
        post_only=True,
    )
    selected = await router.best_venue(order)
    assert selected is a1


@pytest.mark.asyncio
async def test_execute_reports_fee_info():
    ob = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    adapter = MockAdapter("a", order_book=ob, taker_fee_bps=12.0)
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", type_="market", qty=1.0)
    res = await router.execute(order)
    assert res["fee_type"] == "taker"
    assert res["fee_bps"] == 12.0


@pytest.mark.asyncio
async def test_router_records_slippage_metric():
    SLIPPAGE.clear()
    ob = {"XYZ": {"bids": [(99.0, 1.0)], "asks": [(100.0, 1.0)]}}
    adapter = MockAdapter(
        "m",
        order_book=ob,
        last_px={"XYZ": 100.0},
        fill_price=101.0,
    )
    router = ExecutionRouter(adapter)
    order = Order(symbol="XYZ", side="buy", type_="limit", qty=1.0, price=100.0)
    await router.execute(order)
    samples = list(SLIPPAGE.collect())[0].samples
    count_sample = [
        s
        for s in samples
        if s.name == "order_slippage_bps_count" and s.labels["symbol"] == "XYZ" and s.labels["side"] == "buy"
    ][0]
    sum_sample = [
        s
        for s in samples
        if s.name == "order_slippage_bps_sum" and s.labels["symbol"] == "XYZ" and s.labels["side"] == "buy"
    ][0]
    assert count_sample.value == 1.0
    assert sum_sample.value == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_est_slippage_and_queue_paths():
    book = {
        "XYZ": {
            "bids": [(99.0, 2.0)],
            "asks": [(100.0, 1.0), (101.0, 1.0)],
        }
    }
    adapter = MockAdapter("m", order_book=book, last_px={"XYZ": 100.0})

    # Taker path slippage
    router_taker = ExecutionRouter(adapter, prefer="taker")
    order_taker = Order(symbol="XYZ", side="buy", type_="market", qty=1.5)
    res_taker = await router_taker.execute(order_taker)
    assert res_taker["est_slippage_bps"] == pytest.approx(33.333, rel=1e-3)

    # Maker path queue position
    router_maker = ExecutionRouter(adapter, prefer="maker")
    order_maker = Order(
        symbol="XYZ", side="buy", type_="limit", qty=0.5, price=99.0, post_only=True
    )
    res_maker = await router_maker.execute(order_maker)
    assert res_maker["queue_position"] == pytest.approx(0.8, rel=1e-3)


def test_slippage_model_ofi_impact():
    df = pd.DataFrame(
        {
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "volume": [1000.0, 1000.0],
            "bid_qty": [1.0, 2.0],
            "ask_qty": [1.0, 1.0],
        }
    )
    df["order_flow_imbalance"] = calc_ofi(df[["bid_qty", "ask_qty"]])
    bar = df.iloc[1]
    model = SlippageModel(volume_impact=0.0, spread_mult=0.0, ofi_impact=0.5)
    price = 100.0
    adj_buy = model.adjust("buy", 1.0, price, bar)
    adj_sell = model.adjust("sell", 1.0, price, bar)
    assert adj_buy == pytest.approx(price + 0.5, rel=1e-9)
    assert adj_sell == pytest.approx(price - 0.5, rel=1e-9)


def test_slippage_model_sources():
    bar = {"bid": 100.0, "ask": 100.2, "volume": 1000.0}
    price = 100.0
    model_bba = SlippageModel(volume_impact=0.0, source="bba", base_spread=0.05)
    adj_bba = model_bba.adjust("buy", 1.0, price, bar)
    assert adj_bba == pytest.approx(100.1, rel=1e-9)

    bar_missing = {"volume": 1000.0}
    model_fallback = SlippageModel(
        volume_impact=0.0, source="bba", base_spread=0.3
    )
    adj_fallback = model_fallback.adjust("buy", 1.0, price, bar_missing)
    assert adj_fallback == pytest.approx(100.15, rel=1e-9)

    model_fixed = SlippageModel(
        volume_impact=0.0, source="fixed_spread", base_spread=0.4
    )
    adj_fixed = model_fixed.adjust("sell", 1.0, price, bar)
    assert adj_fixed == pytest.approx(99.8, rel=1e-9)

    bar_nan = {"bid": float("nan"), "ask": float("nan"), "volume": 1000.0}
    model_nan = SlippageModel(volume_impact=0.0, source="bba", base_spread=0.3)
    adj_nan = model_nan.adjust("sell", 1.0, price, bar_nan)
    assert adj_nan == pytest.approx(99.85, rel=1e-9)

def test_slippage_helpers():
    asks = [(100.0, 1.0), (101.0, 1.0)]
    bps = impact_by_depth("buy", 1.5, asks)
    assert bps == pytest.approx(33.333, rel=1e-3)

    bids = [(99.0, 2.0)]
    pos = queue_position(0.5, bids)
    assert pos == pytest.approx(0.8, rel=1e-3)
