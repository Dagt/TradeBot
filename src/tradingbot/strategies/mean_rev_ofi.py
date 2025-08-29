import numpy as np
import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import calc_ofi, returns


class MeanRevOFI(Strategy):
    """Mean reversion strategy based on OFI z-score and volatility.

    Generates inverse signals: when buying pressure (positive OFI z-score)
    exceeds a threshold under low volatility, a ``sell`` signal is emitted and
    vice versa.

    Parameters
    ----------
    ofi_window : int, optional
        Lookback window for the OFI rolling statistics, by default ``5``.
    zscore_threshold : float, optional
        Absolute z-score required to trigger a trade, by default ``1.0``.
    vol_window : int, optional
        Window for the volatility estimation of log returns, by default ``20``.
    vol_threshold : float, optional
        Maximum volatility allowed to take a position, by default ``0.01``.
    tp_bps : float, optional
        Take profit in basis points. When reached the position is closed,
        by default ``30.0``.
    sl_bps : float, optional
        Stop loss in basis points. When reached the position is closed,
        by default ``40.0``.
    max_hold_bars : int, optional
        Maximum number of bars to hold a position, by default ``20``.
    """

    name = "mean_rev_ofi"

    def __init__(
        self,
        ofi_window: int = 5,
        zscore_threshold: float = 1.0,
        vol_window: int = 20,
        vol_threshold: float = 0.01,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
        *,
        config_path: str | None = None,
    ) -> None:
        params = load_params(config_path)
        self.ofi_window = int(params.get("ofi_window", ofi_window))
        self.zscore_threshold = float(params.get("zscore_threshold", zscore_threshold))
        self.vol_window = int(params.get("vol_window", vol_window))
        self.vol_threshold = float(params.get("vol_threshold", vol_threshold))
        self.tp_bps = float(params.get("tp_bps", tp_bps))
        self.sl_bps = float(params.get("sl_bps", sl_bps))
        self.max_hold_bars = int(params.get("max_hold_bars", max_hold_bars))
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None

        if self.pos_side != 0:
            self.hold_bars += 1
            assert self.entry_price is not None and last_close is not None
            pnl_bps = (
                (last_close - self.entry_price)
                / self.entry_price
                * 10000
                * self.pos_side
            )
            if (
                pnl_bps >= self.tp_bps
                or pnl_bps <= -self.sl_bps
                or self.hold_bars >= self.max_hold_bars
            ):
                side = "sell" if self.pos_side > 0 else "buy"
                self.pos_side = 0
                self.entry_price = None
                self.hold_bars = 0
                return Signal(side, 1.0)
            return None

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
            return None

        if zscore > self.zscore_threshold:
            self.pos_side = -1
            self.entry_price = last_close
            self.hold_bars = 0
            return Signal("sell", 1.0)
        if zscore < -self.zscore_threshold:
            self.pos_side = 1
            self.entry_price = last_close
            self.hold_bars = 0
            return Signal("buy", 1.0)
        return None
