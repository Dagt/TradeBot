import pandas as pd
from .base import Strategy, Signal
from ..data.features import rsi, order_flow_imbalance

class MeanReversion(Strategy):
    """RSI mean-reversion strategy.

    Parameters are supplied via ``**kwargs`` to allow dynamic
    construction from configuration sources.

    Args:
        rsi_n (int): Window for RSI calculation. Default ``14``.
        upper (float): RSI level considered overbought. Default ``70``.
        lower (float): RSI level considered oversold. Default ``30``.
    """

    name = "mean_reversion"

    def __init__(self, **kwargs):
        self.rsi_n = int(kwargs.get("rsi_n", 14))
        self.upper = float(kwargs.get("upper", 70.0))
        self.lower = float(kwargs.get("lower", 30.0))

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
