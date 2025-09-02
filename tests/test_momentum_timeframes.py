import pandas as pd
from tradingbot.strategies.momentum import Momentum


def test_momentum_signal_1m():
    df = pd.DataFrame({"close": [1, 1, 1, 2, 1, 2], "volume": [1] * 6})
    strat = Momentum(rsi_n=3, vol_window=4, min_volume=0, min_volatility=0)
    sig = strat.on_bar({"window": df, "timeframe": "1m"})
    assert sig is not None and sig.side == "buy"


def test_momentum_signal_15m():
    df = pd.DataFrame({"close": [1, 1, 1, 2, 1], "volume": [1] * 5})
    strat = Momentum(rsi_n=3, vol_window=4, min_volume=0, min_volatility=0)
    sig = strat.on_bar({"window": df, "timeframe": "15m"})
    assert sig is not None and sig.side == "buy"
