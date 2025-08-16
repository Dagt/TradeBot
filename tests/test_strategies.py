import pandas as pd
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.strategies.triangular_arb import TriangularArb
from tradingbot.strategies.order_flow import OrderFlow


def test_breakout_vol_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutVol(ema_n=2, atr_n=2, mult=1.0)

    sig_buy = strat.on_bar({"window": breakout_df_buy})
    assert sig_buy.side == "buy"

    sig_sell = strat.on_bar({"window": breakout_df_sell})
    assert sig_sell.side == "sell"


def test_triangular_arb_signals():
    strat = TriangularArb(threshold=0.01)
    sig = strat.on_bar({"prices": {"bq": 10.0, "mq": 20.0, "mb": 1.8}})
    assert sig and sig.side == "b->m"


def test_order_flow_signals():
    df_buy = pd.DataFrame({
        "bid_qty": [1, 2, 3, 5],
        "ask_qty": [5, 4, 3, 2],
    })
    df_sell = pd.DataFrame({
        "bid_qty": [5, 4, 3, 2],
        "ask_qty": [1, 2, 3, 5],
    })
    strat = OrderFlow(window=3, buy_threshold=1.0, sell_threshold=1.0)
    sig_buy = strat.on_bar({"window": df_buy})
    assert sig_buy.side == "buy"
    sig_sell = strat.on_bar({"window": df_sell})
    assert sig_sell.side == "sell"
