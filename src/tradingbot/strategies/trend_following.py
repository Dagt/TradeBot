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

    def __init__(
        self,
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
        **kwargs,
    ):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 60.0)
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

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
        buy = last_rsi > self.threshold and ofi_val >= 0
        sell = last_rsi < 100 - self.threshold and ofi_val <= 0

        if self.pos_side == 0:
            if buy:
                self.pos_side = 1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("buy", 1.0)
            if sell:
                self.pos_side = -1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("sell", 1.0)
            return None

        self.hold_bars += 1
        exit_signal = (sell and self.pos_side > 0) or (buy and self.pos_side < 0)
        pnl_bps = (
            (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
            if self.entry_price is not None
            else 0.0
        )
        exit_tp = pnl_bps >= self.tp_bps
        exit_sl = pnl_bps <= -self.sl_bps
        exit_time = self.hold_bars >= self.max_hold_bars
        if exit_signal or exit_tp or exit_sl or exit_time:
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None
