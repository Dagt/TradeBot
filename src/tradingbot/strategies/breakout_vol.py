import pandas as pd

from .base import Strategy, Signal


class BreakoutVol(Strategy):
    """Volatility breakout strategy using rolling standard deviation.

    Parameters are supplied via ``**kwargs`` and stored as attributes.  The
    strategy computes a rolling mean and standard deviation of the close price
    and generates a ``buy`` signal when price breaks above ``mean + mult *
    std``.  A ``sell`` signal is produced for a break below ``mean - mult *
    std``.

    Parameters
    ----------
    lookback : int, optional
        Window size for the rolling statistics, default ``20``.
    mult : float, optional
        Multiplier applied to the standard deviation, default ``2``.
    """

    name = "breakout_vol"

    def __init__(self, **kwargs):
        self.lookback = kwargs.get("lookback", 20)
        self.mult = kwargs.get("mult", 2.0)

    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.lookback + 1:
            return None
        closes = df["close"]
        mean = closes.rolling(self.lookback).mean().iloc[-1]
        std = closes.rolling(self.lookback).std().iloc[-1]
        last = closes.iloc[-1]
        upper = mean + self.mult * std
        lower = mean - self.mult * std
        if last > upper:
            return Signal("buy", 1.0)
        if last < lower:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
