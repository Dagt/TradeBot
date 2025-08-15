from tradingbot.strategies.breakout_atr import BreakoutATR


def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0)

    sig_buy = strat.on_bar({"window": breakout_df_buy})
    assert sig_buy.side == "buy"

    sig_sell = strat.on_bar({"window": breakout_df_sell})
    assert sig_sell.side == "sell"
