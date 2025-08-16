import pandas as pd
from .base import Strategy, Signal
from ..data.features import keltner_channels

class BreakoutVol(Strategy):
    """Volatility breakout strategy using Keltner Channels.

    Parameters are provided via ``**kwargs`` making the class easy to
    configure programmatically.

    Args:
        ema_n (int): Period for the EMA used in the middle of the
            Keltner Channel. Default ``20``.
        atr_n (int): Period for the ATR. Default ``14``.
        mult (float): Multiplier for the ATR band width. Default ``1.5``.
    """

    name = "breakout_vol"

    def __init__(self, **kwargs):
        self.ema_n = int(kwargs.get("ema_n", 20))
        self.atr_n = int(kwargs.get("atr_n", 14))
        self.mult = float(kwargs.get("mult", 1.5))

    def on_bar(self, bar: dict) -> Signal | None:
        # bar should include a small rolling window (as dict of lists) or a pandas row with context
        df: pd.DataFrame = bar["window"]  # expects columns: open, high, low, close, volume
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = df["close"].iloc[-1]
        if last_close > upper.iloc[-1]:
            return Signal("buy", 1.0)
        if last_close < lower.iloc[-1]:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
