import pytest
from tradingbot.backtesting.engine import SlippageModel
from tradingbot.execution.paper import PaperAdapter


@pytest.mark.asyncio
async def test_average_slippage_matches_backtest():
    model = SlippageModel(
        volume_impact=0.1,
        spread_mult=1.0,
        ofi_impact=0.0,
        source="bba",
        base_spread=0.0,
        pct=0.0001,
    )
    adapter = PaperAdapter(slippage_model=model, slip_bps_per_qty=0.0)
    adapter.state.cash = 1_000_000.0
    adapter.update_last_price("BTC/USDT", 100.0)

    bars = [
        {"bid": 99.5, "ask": 100.5, "bid_size": 10.0, "ask_size": 10.0, "volume": 1000.0},
        {"bid": 99.0, "ask": 101.0, "bid_size": 20.0, "ask_size": 20.0, "volume": 1000.0},
    ]
    orders = [
        ("buy", 2.0, 100.0),
        ("buy", 4.0, 100.0),
    ]

    slips_bt = []
    slips_pp = []
    for (side, qty, price), bar in zip(orders, bars):
        res = await adapter.place_order("BTC/USDT", side, "market", qty, book=bar)
        slips_pp.append(res["slippage_bps"])
        adj_price, _, _ = model.fill(side, qty, price, bar)
        slips_bt.append((adj_price - price) / price * 10000)

    avg_bt = sum(slips_bt) / len(slips_bt)
    avg_pp = sum(slips_pp) / len(slips_pp)
    assert avg_pp == pytest.approx(avg_bt)
