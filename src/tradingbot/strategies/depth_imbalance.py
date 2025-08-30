from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import depth_imbalance


PARAM_INFO = {
    "window": "Ventana para promediar el desequilibrio",
    "threshold": "Umbral de desequilibrio para operar",
    "max_duration": "Máxima duración de la posición",
}


class DepthImbalance(Strategy):
    """Depth Imbalance strategy.

    Computes the mean depth imbalance over a rolling window and issues
    directional signals when the average exceeds ``threshold``.
    """

    name = "depth_imbalance"

    def __init__(
        self,
        window: int = 3,
        threshold: float = 0.2,
        tp: float | None = None,
        sl: float | None = None,
        max_duration: pd.Timedelta | int | float | str | None = None,
    ):
        self.window = window
        self.threshold = threshold
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

        di_series = depth_imbalance(df[list(needed)])
        di_mean = di_series.iloc[-self.window :].mean()
        if di_mean > self.threshold:
            self.pos_side = "buy"
            self.entry_price = price
            self.entry_time = now
            return Signal("buy", 1.0)
        if di_mean < -self.threshold:
            self.pos_side = "sell"
            self.entry_price = price
            self.entry_time = now
            return Signal("sell", 1.0)
        return None
