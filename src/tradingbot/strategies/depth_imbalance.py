from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import depth_imbalance


PARAM_INFO = {
    "window": "Ventana para promediar el desequilibrio",
    "threshold": "Umbral de desequilibrio para operar",
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
        **kwargs,
    ):
        self.window = window
        self.threshold = threshold
        self.risk_service = kwargs.get("risk_service")

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None

        price: float | None = None
        if len(df):
            if "close" in df.columns:
                last_px = df["close"].iloc[-1]
                if pd.notna(last_px):
                    price = float(last_px)
            elif "price" in df.columns:
                last_px = df["price"].iloc[-1]
                if pd.notna(last_px):
                    price = float(last_px)
        if price is None:
            return None

        di_series = depth_imbalance(df[list(needed)])
        di_mean = di_series.iloc[-self.window :].mean()
        if di_mean > self.threshold:
            side = "buy"
        elif di_mean < -self.threshold:
            side = "sell"
        else:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        return self.finalize_signal(bar, price, sig)
