import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi


class OrderFlow(Strategy):
    """Order Flow Imbalance strategy.

    Calculates the mean Order Flow Imbalance (OFI) over a rolling window and
    issues buy/sell signals when the mean exceeds the configured thresholds.
    """

    name = "order_flow"

    def __init__(
        self,
        window: int = 3,
        buy_threshold: float = 1.0,
        sell_threshold: float = 1.0,
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        ofi_series = calc_ofi(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window:].mean()
        buy = ofi_mean > self.buy_threshold
        sell = ofi_mean < -self.sell_threshold
        price = None
        if {"close"}.issubset(df.columns):
            price = float(df["close"].iloc[-1])

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
        exit_tp = exit_sl = False
        if price is not None and self.entry_price is not None:
            pnl_bps = (
                (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
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
