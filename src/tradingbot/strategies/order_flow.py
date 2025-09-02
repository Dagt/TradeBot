import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi
from ..data.features import returns


PARAM_INFO = {
    "window": "Ventana para promediar el OFI",
    "buy_threshold": "Multiplicador del umbral de compra (z-score)",
    "sell_threshold": "Multiplicador del umbral de venta (z-score)",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class OrderFlow(Strategy):
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
        min_volatility: float | None = None,
        **kwargs,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_volatility = min_volatility
        self.buy_threshold_bps = 0.0
        self.sell_threshold_bps = 0.0
        self._min_volatility = 0.0
        self.risk_service = kwargs.get("risk_service")

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
        if price is None:
            return None

        price_col = "close" if "close" in df.columns else None
        vol_bps = 0.0
        if price_col:
            closes = df[price_col]
            rets = closes.pct_change().dropna()
            vol = (
                rets.rolling(self.window).std().iloc[-1]
                if len(rets) >= self.window
                else 0.0
            )
            vol_bps = vol * 10000
            if self.min_volatility is None:
                vol_series = rets.rolling(self.window).std().dropna()
                window = min(len(vol_series), self.window * 5)
                if window >= self.window:
                    self._min_volatility = float(
                        (vol_series * 10000).rolling(window).quantile(0.2).iloc[-1]
                    )
                else:
                    self._min_volatility = 0.0
            else:
                self._min_volatility = self.min_volatility
        if vol_bps < self._min_volatility:
            return None

        ofi_series = calc_ofi(df[list(needed)])
        rolling_mean = ofi_series.rolling(self.window).mean()
        rolling_std = ofi_series.rolling(self.window).std(ddof=0).replace(0, pd.NA)
        ofi_z = ((ofi_series - rolling_mean) / rolling_std).iloc[-1]
        if pd.isna(ofi_z):
            return None

        ofi_bps = ofi_z * vol_bps
        self.buy_threshold_bps = self.buy_threshold * vol_bps
        self.sell_threshold_bps = self.sell_threshold * vol_bps

        if ofi_bps > self.buy_threshold_bps:
            side = "buy"
        elif ofi_bps < -self.sell_threshold_bps:
            side = "sell"
        else:
            return self.finalize_signal(bar, price, None)

        strength = 1.0
        sig = Signal(side, strength)
        return self.finalize_signal(bar, price, sig)
