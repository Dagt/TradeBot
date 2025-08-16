import pandas as pd
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow


def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0)

    sig_buy = strat.on_bar({"window": breakout_df_buy})
    assert sig_buy.side == "buy"

    sig_sell = strat.on_bar({"window": breakout_df_sell})
    assert sig_sell.side == "sell"


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
