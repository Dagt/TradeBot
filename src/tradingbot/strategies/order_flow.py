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
        tp: float | None = None,
        sl: float | None = None,
        max_duration: pd.Timedelta | int | float | str | None = None,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.tp = tp
        self.sl = sl
        self.max_duration = pd.Timedelta(max_duration) if max_duration else None
        self.pos_side: str | None = None
        self.entry_price: float | None = None
        self.entry_time: pd.Timestamp | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None

        price = bar.get("close")
        now = pd.Timestamp(bar.get("ts") or bar.get("timestamp") or pd.Timestamp.utcnow())

        if self.pos_side and self.entry_price is not None:
            pnl = (
                price - self.entry_price
                if self.pos_side == "buy"
                else self.entry_price - price
            ) if price is not None else None
            pct = pnl / self.entry_price if pnl is not None else None
            if pct is not None:
                if self.tp is not None and pct >= self.tp:
                    side = self.pos_side
                    self.pos_side = None
                    self.entry_price = None
                    self.entry_time = None
                    return Signal(side, 0.0)
                if self.sl is not None and pct <= -self.sl:
                    side = self.pos_side
                    self.pos_side = None
                    self.entry_price = None
                    self.entry_time = None
                    return Signal(side, 0.0)
            if (
                self.max_duration is not None
                and self.entry_time is not None
                and now - self.entry_time >= self.max_duration
            ):
                side = self.pos_side
                self.pos_side = None
                self.entry_price = None
                self.entry_time = None
                return Signal(side, 0.0)

        ofi_series = calc_ofi(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window:].mean()
        if ofi_mean > self.buy_threshold:
            self.pos_side = "buy"
            self.entry_price = price
            self.entry_time = now
            return Signal("buy", 1.0)
        if ofi_mean < -self.sell_threshold:
            self.pos_side = "sell"
            self.entry_price = price
            self.entry_time = now
            return Signal("sell", 1.0)
        return None
