import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import order_flow_imbalance


class OrderFlow(Strategy):
    """Order Flow Imbalance strategy.

    Calculates the mean Order Flow Imbalance (OFI) over a rolling window and
    issues buy/sell signals when the mean exceeds the configured thresholds.
    """

    name = "order_flow"

    def __init__(self, window: int = 3, buy_threshold: float = 1.0, sell_threshold: float = 1.0):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        ofi_series = order_flow_imbalance(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window:].mean()
        if ofi_mean > self.buy_threshold:
            return Signal("buy", 1.0)
        if ofi_mean < -self.sell_threshold:
            return Signal("sell", 1.0)
        return Signal("flat", 0.0)
