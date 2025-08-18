import pandas as pd
from tradingbot.strategies.mean_reversion import MeanReversion, generate_signals

def test_mean_reversion_on_bar_signals():
    df_down = pd.DataFrame({"close": list(range(20, 0, -1))})
    df_up = pd.DataFrame({"close": list(range(1, 21))})
    strat = MeanReversion(rsi_n=5, upper=70, lower=30)
    sig_buy = strat.on_bar({"window": df_down})
    sig_sell = strat.on_bar({"window": df_up})
    assert sig_buy.side == "buy"
    assert sig_sell.side == "sell"

def test_mean_reversion_generate_signals():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 3, 2, 1, 2, 3]})
    params = {"window": 3, "threshold": 0.5, "position_size": 1}
    res = generate_signals(df, params)
    assert {"signal", "position", "stop_loss", "take_profit", "fee", "slippage"} <= set(res.columns)
    assert len(res) == len(df)
