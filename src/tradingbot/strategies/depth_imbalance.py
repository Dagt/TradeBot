from __future__ import annotations

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import depth_imbalance
from ..filters.liquidity import LiquidityFilterManager


liquidity = LiquidityFilterManager()


PARAM_INFO = {
    "window": "Ventana para estimar el desequilibrio",
    "percentile": "Percentil para derivar el umbral dinámico",
}


class DepthImbalance(Strategy):
    """Depth Imbalance strategy.

    Computes the depth imbalance over a rolling window and issues
    directional signals when the latest value exceeds a dynamic
    threshold derived from the specified percentile.
    """

    name = "depth_imbalance"

    def __init__(
        self,
        window: int = 3,
        percentile: float = 80.0,
        **kwargs,
    ):
        self.window = window
        self.percentile = percentile
        self.risk_service = kwargs.get("risk_service")

    @record_signal_metrics(liquidity)
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
            price = bar.get("close") or bar.get("price")
        if price is None:
            return None

        if self.risk_service is not None and getattr(self, "trade", None):
            self.risk_service.update_trailing(self.trade, price)
            self.trade["_trail_done"] = True
            decision = self.risk_service.manage_position(
                {**self.trade, "current_price": price}, None
            )
            if decision == "close":
                side = "sell" if self.trade.get("side") == "buy" else "buy"
                close_sig = Signal(side, 1.0)
                self.trade = None
                return self.finalize_signal(bar, price, close_sig)

        di_series = depth_imbalance(df[list(needed)])
        window_di = di_series.iloc[-self.window :]
        threshold = window_di.abs().quantile(self.percentile / 100)
        last_di = di_series.iloc[-1]
        if last_di > threshold:
            side = "buy"
        elif last_di < -threshold:
            side = "sell"
        else:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(strength, price)
            atr_val = bar.get("atr") or bar.get("volatility") or 0.0
            stop = self.risk_service.initial_stop(price, side, atr_val)
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
            }
            self.risk_service.update_trailing(self.trade, price)
            self.trade["_trail_done"] = True
        return self.finalize_signal(bar, price, sig)
