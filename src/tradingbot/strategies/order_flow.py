import pandas as pd
from .base import StatefulStrategy, Signal, record_signal_metrics
from ..data.features import calc_ofi


class OrderFlow(StatefulStrategy):
    """Order Flow Imbalance strategy.

    Calculates the mean Order Flow Imbalance (OFI) over a rolling window and
    issues buy/sell signals when the mean exceeds the configured thresholds.
    """

    name = "order_flow"

    def __init__(
        self,
        window: int = 3,
        buy_threshold: float = 1.0,
        sell_threshold: float = 1.0,
        *,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold_bars)
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        price = float(df.get("close", df[list(needed)].mean(axis=1).iloc[-1]))
        exit_sig = self.check_exit(price)
        if exit_sig:
            return exit_sig
        ofi_series = calc_ofi(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window:].mean()
        if self.pos_side is None:
            if ofi_mean > self.buy_threshold:
                return self.open_position("buy", price)
            if ofi_mean < -self.sell_threshold:
                return self.open_position("sell", price)
        return None
