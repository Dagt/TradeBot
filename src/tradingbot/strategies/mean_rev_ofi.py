import numpy as np
import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi, returns


class MeanRevOFI(Strategy):
    """Mean reversion strategy based on OFI z-score and volatility.

    Generates inverse signals: when buying pressure (positive OFI z-score)
    exceeds a threshold under low volatility, a ``sell`` signal is emitted and
    vice versa.

    Parameters
    ----------
    ofi_window : int, optional
        Lookback window for the OFI rolling statistics, by default ``20``.
    zscore_threshold : float, optional
        Absolute z-score required to trigger a trade, by default ``1.0``.
    vol_window : int, optional
        Window for the volatility estimation of log returns, by default ``20``.
    vol_threshold : float, optional
        Maximum volatility allowed to take a position, by default ``0.01``.
    """

    name = "mean_rev_ofi"

    def __init__(
        self,
        ofi_window: int = 20,
        zscore_threshold: float = 1.0,
        vol_window: int = 20,
        vol_threshold: float = 0.01,
    ) -> None:
        self.ofi_window = ofi_window
        self.zscore_threshold = zscore_threshold
        self.vol_window = vol_window
        self.vol_threshold = vol_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty", "close"}
        min_len = max(self.ofi_window, self.vol_window) + 1
        if not needed.issubset(df.columns) or len(df) < min_len:
            return None

        ofi_series = calc_ofi(df[["bid_qty", "ask_qty"]])
        rolling_mean = ofi_series.rolling(self.ofi_window).mean()
        rolling_std = ofi_series.rolling(self.ofi_window).std(ddof=0).replace(0, np.nan)
        zscore = ((ofi_series - rolling_mean) / rolling_std).iloc[-1]

        vol = returns(df).rolling(self.vol_window).std().iloc[-1]

        if pd.isna(zscore) or pd.isna(vol) or vol >= self.vol_threshold:
            return Signal("flat", 0.0)

        if zscore > self.zscore_threshold:
            return Signal("sell", 1.0)
        if zscore < -self.zscore_threshold:
            return Signal("buy", 1.0)
        return Signal("flat", 0.0)
