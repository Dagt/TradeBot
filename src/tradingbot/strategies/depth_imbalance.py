from __future__ import annotations

import pandas as pd

from .base import StatefulStrategy, Signal, record_signal_metrics
from ..data.features import depth_imbalance


class DepthImbalance(StatefulStrategy):
    """Depth Imbalance strategy.

    Computes the mean depth imbalance over a rolling window and issues
    directional signals when the average exceeds ``threshold``.
    """

    name = "depth_imbalance"

    def __init__(
        self,
        window: int = 3,
        threshold: float = 0.2,
        *,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold_bars)
        self.window = window
        self.threshold = threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        price = float(df.get("close", df[list(needed)].mean(axis=1).iloc[-1]))
        exit_sig = self.check_exit(price)
        if exit_sig:
            return exit_sig
        di_series = depth_imbalance(df[list(needed)])
        di_mean = di_series.iloc[-self.window :].mean()
        if self.pos_side is None:
            if di_mean > self.threshold:
                return self.open_position("buy", price)
            if di_mean < -self.threshold:
                return self.open_position("sell", price)
        return None
