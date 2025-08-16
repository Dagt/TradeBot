import pandas as pd
from .base import Strategy, Signal
from ..data.features import rsi, order_flow_imbalance

class Momentum(Strategy):
    """Relative Strength Index (RSI) momentum strategy.

    Parameters are passed via ``**kwargs`` so the class can be
    instantiated dynamically from configuration files or the CLI.

    Args:
        rsi_n (int): Period for the RSI calculation. Default ``14``.
        rsi_threshold (float): RSI value above which a ``buy`` signal is
            produced (and symmetrically for ``sell``).  Default ``60``.
    """

    name = "momentum"

    def __init__(self, **kwargs):
        self.rsi_n = int(kwargs.get("rsi_n", 14))
        self.threshold = float(kwargs.get("rsi_threshold", 60.0))

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
