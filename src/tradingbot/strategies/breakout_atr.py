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
        min_bars_between_trades: int = 1,
        *,
        config_path: str | None = None,
    ):
        params = load_params(config_path)
        self.ema_n = int(params.get("ema_n", ema_n))
        self.atr_n = int(params.get("atr_n", atr_n))
        self.mult = float(params.get("mult", mult))
        mbbt = params.get("min_bars_between_trades", min_bars_between_trades)
        self.min_bars_between_trades = max(int(mbbt), 1)
        self._last_trade_idx: int | None = None
        self._last_trade_side: str | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        # bar should include a small rolling window (as dict of lists)
        # or a pandas row with context
        df: pd.DataFrame = bar["window"]
        # expects columns: open, high, low, close, volume
        if len(df) < max(self.ema_n, self.atr_n) + 2:
            return None
        upper, lower = keltner_channels(df, self.ema_n, self.atr_n, self.mult)
        last_close = df["close"].iloc[-1]
        current_idx = len(df) - 1

        sig: Signal
        if last_close > upper.iloc[-1]:
            sig = Signal("buy", 1.0)
        elif last_close < lower.iloc[-1]:
            sig = Signal("sell", 1.0)
        else:
            sig = Signal("flat", 0.0)

        if sig.side in {"buy", "sell"}:
            if (
                self._last_trade_idx is not None
                and self._last_trade_side is not None
                and sig.side != self._last_trade_side
                and current_idx - self._last_trade_idx
                < self.min_bars_between_trades
            ):
                return Signal("flat", 0.0)
            self._last_trade_idx = current_idx
            self._last_trade_side = sig.side

        return sig
