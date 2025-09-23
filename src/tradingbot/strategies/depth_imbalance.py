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
            trade = self.trade
            self.risk_service.update_trailing(trade, price)
            is_dict = isinstance(trade, dict)
            _missing = object()
            if is_dict:
                trail_prev = trade.get("_trail_done") if "_trail_done" in trade else _missing
            else:
                trail_prev = (
                    getattr(trade, "_trail_done") if hasattr(trade, "_trail_done") else _missing
                )
            if is_dict:
                trade["_trail_done"] = True
                had_price = "current_price" in trade
                prev_price = trade.get("current_price") if had_price else None
                trade["current_price"] = price
            else:
                setattr(trade, "_trail_done", True)
                had_price = hasattr(trade, "current_price")
                prev_price = getattr(trade, "current_price") if had_price else None
                setattr(trade, "current_price", price)
            try:
                decision = self.risk_service.manage_position(trade, None)
            finally:
                if is_dict:
                    if had_price:
                        trade["current_price"] = prev_price
                    else:
                        trade.pop("current_price", None)
                    if trail_prev is _missing:
                        trade.pop("_trail_done", None)
                    else:
                        trade["_trail_done"] = trail_prev
                else:
                    if had_price:
                        setattr(trade, "current_price", prev_price)
                    else:
                        if hasattr(trade, "current_price"):
                            delattr(trade, "current_price")
                    if trail_prev is _missing:
                        if hasattr(trade, "_trail_done"):
                            delattr(trade, "_trail_done")
                    else:
                        setattr(trade, "_trail_done", trail_prev)
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
            trade = self.trade
            self.risk_service.update_trailing(trade, price)
            if isinstance(trade, dict):
                trade["_trail_done"] = True
            else:
                setattr(trade, "_trail_done", True)
        return self.finalize_signal(bar, price, sig)
