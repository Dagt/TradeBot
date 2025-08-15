import pandas as pd
from .base import Strategy, Signal
from ..data.features import rsi, order_flow_imbalance

class Momentum(Strategy):
    name = "momentum"

    def __init__(self, rsi_n: int = 14, rsi_threshold: float = 60.0):
        self.rsi_n = rsi_n
        self.threshold = rsi_threshold

    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = order_flow_imbalance(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if last_rsi > self.threshold and ofi_val >= 0:
            return Signal("buy", 1.0)
        if last_rsi < 100 - self.threshold and ofi_val <= 0:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
