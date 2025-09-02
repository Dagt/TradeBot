import numpy as np
import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import calc_ofi, returns
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager


PARAM_INFO = {
    "ofi_window": "Ventana para estadísticos de OFI",
    "zscore_threshold": "Z-score absoluto requerido",
    "vol_window": "Ventana para volatilidad de retornos",
    "vol_threshold": "Percentil para volatilidad máxima en bps",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


liquidity = LiquidityFilterManager()


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
        Window for the volatility estimation of log returns at ``3m`` timeframe,
        by default ``20``.
    vol_threshold : float, optional
        Percentile (0-1) of volatility in bps used as maximum allowed level,
        by default ``0.8``.
    """

    name = "mean_rev_ofi"

    def __init__(
        self,
        ofi_window: int = 5,
        zscore_threshold: float = 1.0,
        vol_window: int = 20,
        vol_threshold: float = 0.8,
        min_volatility: float = 0.0,
        *,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        params = {**load_params(config_path), **kwargs}
        self.risk_service = params.pop("risk_service", None)
        self.ofi_window = int(params.get("ofi_window", ofi_window))
        self.zscore_threshold = float(params.get("zscore_threshold", zscore_threshold))
        self.vol_window = int(params.get("vol_window", vol_window))
        self.vol_threshold = float(params.get("vol_threshold", vol_threshold))
        self.vol_threshold_bps = 0.0
        self.min_volatility = float(params.get("min_volatility", min_volatility))
        self._rq = RollingQuantileCache()

    def _tf_minutes(self, timeframe: str) -> float:
        unit = timeframe[-1].lower()
        value = float(timeframe[:-1])
        factors = {"s": 1 / 60, "m": 1, "h": 60, "d": 1440}
        return value * factors.get(unit, 0)

    def _scaled_vol_window(self, timeframe: str | None) -> int:
        if timeframe is None:
            return self.vol_window
        tf_min = self._tf_minutes(timeframe)
        base_min = 3.0
        return max(1, int(round(self.vol_window * base_min / tf_min)))

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None


        needed = {"bid_qty", "ask_qty", "close"}
        vol_window = self._scaled_vol_window(bar.get("timeframe"))
        min_len = max(self.ofi_window, vol_window) + 1
        if not needed.issubset(df.columns) or len(df) < min_len:
            return None

        ofi_series = calc_ofi(df[["bid_qty", "ask_qty"]])
        rolling_mean = ofi_series.rolling(self.ofi_window).mean()
        rolling_std = ofi_series.rolling(self.ofi_window).std(ddof=0).replace(0, np.nan)
        zscore = ((ofi_series - rolling_mean) / rolling_std).iloc[-1]

        vol_series = returns(df).rolling(vol_window).std()
        vol_bps_series = vol_series * 10000
        vol_bps = float(vol_bps_series.iloc[-1])
        symbol = bar.get("symbol", "")
        rq = self._rq.get(
            symbol,
            "vol_bps",
            window=vol_window * 5,
            q=self.vol_threshold,
            min_periods=vol_window,
        )
        self.vol_threshold_bps = float(rq.update(vol_bps))
        if pd.isna(self.vol_threshold_bps):
            self.vol_threshold_bps = float("inf")

        if (
            pd.isna(zscore)
            or pd.isna(vol_bps)
            or vol_bps >= self.vol_threshold_bps
            or vol_bps < self.min_volatility
        ):
            return None

        if zscore > self.zscore_threshold:
            side = "sell"
        elif zscore < -self.zscore_threshold:
            side = "buy"
        else:
            return self.finalize_signal(bar, last_close or 0.0, None)
        strength = 1.0
        sig = Signal(side, strength)
        return self.finalize_signal(bar, last_close or 0.0, sig)
