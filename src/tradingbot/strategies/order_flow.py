import pandas as pd

from .base import Strategy, Signal, record_signal_metrics
from ..data.features import calc_ofi


PARAM_INFO = {
    "window": "Ventana para promediar el OFI",
    "buy_threshold": "Umbral de compra para OFI",
    "sell_threshold": "Umbral de venta para OFI",
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
        min_volatility: float = 0.0,
        **kwargs,
    ):
        self.window = window
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_volatility = min_volatility
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
        if vol_bps < self.min_volatility:
            return None

        ofi_series = calc_ofi(df[list(needed)])
        ofi_mean = ofi_series.iloc[-self.window :].mean()
        if ofi_mean > self.buy_threshold:
            side = "buy"
        elif ofi_mean < -self.sell_threshold:
            side = "sell"
        else:
            return self.finalize_signal(bar, price or 0.0, None)
        if price is None:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        return self.finalize_signal(bar, price, sig)
