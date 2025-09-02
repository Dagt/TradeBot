import pandas as pd
from tradingbot.strategies.momentum import Momentum

def test_momentum_sets_limit_price():
    df = pd.DataFrame({"close": [1, 2, 1, 2]})
    strat = Momentum(rsi_n=2)
    sig = strat.on_bar({"window": df, "close": df["close"].iloc[-1], "volatility": 0.0})
    assert sig is not None and sig.side == "buy"
    assert sig.limit_price == df["close"].iloc[-1]
