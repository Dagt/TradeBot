import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, order_flow_imbalance

class MeanReversion(Strategy):
    """RSI based mean reversion strategy.

    Parameters are accepted through ``**kwargs`` for easy configuration.

    Parameters
    ----------
    rsi_n : int, optional
        Lookback window for the RSI calculation, by default ``14``.
    upper : float, optional
        Upper RSI level above which a ``sell`` signal is triggered, by default
        ``70``.
    lower : float, optional
        Lower RSI level below which a ``buy`` signal is triggered, by default
        ``30``.
    """

    name = "mean_reversion"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.upper = kwargs.get("upper", 70.0)
        self.lower = kwargs.get("lower", 30.0)

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = order_flow_imbalance(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if last_rsi > self.upper and ofi_val <= 0:
            return Signal("sell", 1.0)
        if last_rsi < self.lower and ofi_val >= 0:
            return Signal("buy", 1.0)
        return Signal("flat", 0.0)
