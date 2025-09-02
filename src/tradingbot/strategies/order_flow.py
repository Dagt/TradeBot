import numpy as np
import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi
from ..utils.rolling_quantile import RollingQuantileCache


PARAM_INFO = {
    "window": "Ventana para estadísticos de OFI",
    "buy_threshold": "Umbral de compra en bps",
    "sell_threshold": "Umbral de venta en bps",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class OrderFlow(Strategy):
    """Order Flow Imbalance strategy.

    Uses the z-score of the Order Flow Imbalance (OFI) scaled by recent
    volatility in basis points.  Buy/sell signals are issued when the resulting
    dynamic value exceeds the configured thresholds.
    """

    name = "order_flow"

    def __init__(
        self,
        window: int = 3,
        buy_threshold: float = 1.0,
        sell_threshold: float = 1.0,
        min_volatility: float | None = None,
        **kwargs,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_volatility = min_volatility
        self.risk_service = kwargs.get("risk_service")
        self._rq = RollingQuantileCache()

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        needed = {"bid_qty", "ask_qty"}
        if not needed.issubset(df.columns) or len(df) < self.window:
            return None
        price = bar.get("close")
        if price is None:
            if "close" in df.columns:
                price = float(df["close"].iloc[-1])
            elif "price" in df.columns:
                price = float(df["price"].iloc[-1])

        vol_bps = float("inf")
        price_col = "close" if "close" in df.columns else None
        if price_col:
            closes = df[price_col]
            returns = closes.pct_change().dropna()
            vol = (
                returns.rolling(self.window).std().iloc[-1]
                if len(returns) >= self.window
                else 0.0
            )
            vol_bps = vol * 10000

            if self.min_volatility is None:
                vol_series = returns.rolling(self.window).std() * 10000
                symbol = bar.get("symbol", "")
                rq = self._rq.get(
                    symbol,
                    "vol_bps",
                    window=self.window * 5,
                    q=0.1,
                    min_periods=self.window,
                )
                val = rq.update(float(vol_series.iloc[-1]) if len(vol_series) else 0.0)
                self.min_volatility = 0.0 if pd.isna(val) else float(val)

        if self.min_volatility is not None and vol_bps < self.min_volatility:
            return None

        ofi_series = calc_ofi(df[list(needed)])
        rolling = ofi_series.rolling(self.window)
        ofi_mean = rolling.mean()
        ofi_std = rolling.std(ddof=0).replace(0, np.nan)
        zscore = ((ofi_series - ofi_mean) / ofi_std).iloc[-1]
        if pd.isna(zscore) or not np.isfinite(vol_bps):
            return None

        ofi_bps = zscore * vol_bps
        if ofi_bps > self.buy_threshold:
            side = "buy"
        elif ofi_bps < -self.sell_threshold:
            side = "sell"
        else:
            return self.finalize_signal(bar, price or 0.0, None)

        if price is None:
            return None
        sig = Signal(side, 1.0)
        return self.finalize_signal(bar, price, sig)
