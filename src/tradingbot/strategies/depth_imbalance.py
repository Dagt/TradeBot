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
        risk_service=None,
    ):
        self.window = window
        self.threshold = threshold
        self.risk_service = risk_service
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None

        price = bar.get("close")
        if self.trade and self.risk_service and price is not None:
            self.risk_service.update_trailing(self.trade, price)
            trade_state = {**self.trade, "current_price": price}
            decision = self.risk_service.manage_position(trade_state)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            if decision in {"scale_in", "scale_out"}:
                self.trade["strength"] = trade_state.get("strength", 1.0)
                return Signal(self.trade["side"], self.trade["strength"])
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
        if self.risk_service and price is not None:
            qty = self.risk_service.calc_position_size(strength, price)
            trade = {"side": side, "entry_price": price, "qty": qty, "strength": strength}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(price, side, atr)
            if atr is not None:
                trade["atr"] = atr
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        return Signal(side, strength)
