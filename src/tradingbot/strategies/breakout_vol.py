import pandas as pd

from .base import Strategy, Signal, record_signal_metrics


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
        self.mult = kwargs.get("mult", 1.5)
        self.tp_bps = kwargs.get("tp_bps", 30.0)
        self.sl_bps = kwargs.get("sl_bps", 40.0)
        self.max_hold_bars = kwargs.get("max_hold_bars", 20)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
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

        if self.pos_side == 0:
            if last > upper:
                self.pos_side = 1
                self.entry_price = last
                self.hold_bars = 0
                return Signal("buy", 1.0)
            if last < lower:
                self.pos_side = -1
                self.entry_price = last
                self.hold_bars = 0
                return Signal("sell", 1.0)
            return None

        self.hold_bars += 1
        assert self.entry_price is not None
        pnl_bps = (last - self.entry_price) / self.entry_price * 10000 * self.pos_side
        if (
            pnl_bps >= self.tp_bps
            or pnl_bps <= -self.sl_bps
            or self.hold_bars >= self.max_hold_bars
        ):
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None
