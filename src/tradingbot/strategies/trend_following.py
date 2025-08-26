import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, calc_ofi


class TrendFollowing(Strategy):
    """RSI based trend following strategy with adaptive strength.

    Signals are generated when the RSI crosses extreme levels.  The returned
    ``strength`` scales up if an existing position is profitable and the new
    signal aligns with it.  Adverse moves reduce the strength and may turn it
    negative to indicate that the position should be reduced or flipped.
    """

    name = "trend_following"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 60.0)
        self._pos_side: str | None = None
        self._entry_price: float | None = None

    def _calc_strength(self, side: str, price: float) -> float:
        if side == "flat":
            self._pos_side = None
            self._entry_price = None
            return 0.0
        strength = 1.0
        if self._pos_side and self._entry_price:
            pnl = (price - self._entry_price) / self._entry_price
            if self._pos_side == "sell":
                pnl = -pnl
            if side == self._pos_side:
                strength += pnl
            else:
                strength = -pnl
        if strength > 0:
            self._pos_side = side
            self._entry_price = price
        else:
            self._pos_side = None
            self._entry_price = None
        return strength

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        price_col = "close" if "close" in df.columns else "price"
        price = float(df[price_col].iloc[-1])
        if last_rsi > self.threshold and ofi_val >= 0:
            strength = self._calc_strength("buy", price)
            return Signal("buy", strength)
        if last_rsi < 100 - self.threshold and ofi_val <= 0:
            strength = self._calc_strength("sell", price)
            return Signal("sell", strength)
        strength = self._calc_strength("flat", price)
        return Signal("flat", strength)
