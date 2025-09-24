import pandas as pd
import pytest

from tradingbot.strategies.momentum import Momentum


def test_momentum_signal_1m():
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 2, 1, 2],
            "high": [1, 1, 1, 2, 1, 2],
            "low": [1, 1, 1, 2, 1, 2],
            "close": [1, 1, 1, 2, 1, 2],
            "volume": [1] * 6,
        }
    )
    strat = Momentum(
        fast_ema=2,
        slow_ema=4,
        rsi_n=3,
        atr_n=3,
        roc_n=1,
        vol_window=4,
        min_volume=0,
        min_volatility=0,
    )
    sig = strat.on_bar({"window": df, "timeframe": "1m", "symbol": "X"})
    assert sig is not None and sig.side == "buy"


@pytest.mark.parametrize("timeframe", ["15m", "1h"])
def test_momentum_signals_on_longer_timeframes(timeframe: str):
    prices = [
        120,
        118,
        116,
        114,
        112,
        110,
        108,
        106,
        104,
        102,
        100,
        99,
        98,
        97,
        96,
        95,
        94,
        94,
        93,
        92,
        150,
    ]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [10] * len(prices),
        }
    )
    strat = Momentum(
        fast_ema=30,
        slow_ema=90,
        rsi_n=30,
        atr_n=30,
        roc_n=3,
        vol_window=30,
        min_volume=0,
        min_volatility=0,
    )
    sig = strat.on_bar({"window": df, "timeframe": timeframe, "symbol": "X"})
    assert sig is not None and sig.side == "buy"
