import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import keltner_channels

class BreakoutATR(Strategy):
    name = "breakout_atr"

    def __init__(
        self,
        ema_n: int = 20,
        atr_n: int = 14,
        mult: float = 1.5,
        *,
        config_path: str | None = None,
    ):
        params = load_params(config_path)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        # bar should include a small rolling window (as dict of lists) or a pandas row with context
        df: pd.DataFrame = bar["window"]  # expects columns: open, high, low, close, volume
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = df["close"].iloc[-1]
        if last_close > upper.iloc[-1]:
            return Signal("buy", 1.0, target_pct=1.0)
        if last_close < lower.iloc[-1]:
            return Signal("sell", 1.0, target_pct=1.0)
        return Signal("flat", 0.0, target_pct=0.0)
