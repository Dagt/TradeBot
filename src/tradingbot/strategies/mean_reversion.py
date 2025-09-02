import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi

PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_threshold": "Umbral para considerar la tendencia fuerte",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class MeanReversion(Strategy):
    """RSI based mean reversion strategy with adaptive strength.

    Generates ``buy`` or ``sell`` signals when the RSI deviates from
    dynamically determined thresholds. Signal strength scales with the
    distance from the threshold.
    """

    name = "mean_reversion"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.trend_ma = kwargs.get("trend_ma", 50)
        self.trend_rsi_n = kwargs.get("trend_rsi_n", 50)
        self.trend_threshold = kwargs.get("trend_threshold", 10.0)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.risk_service = kwargs.get("risk_service")

    def auto_threshold(self, rsi_series: pd.Series) -> tuple[float, float]:
        """Derive upper and lower RSI bounds from recent variability."""

        dev = rsi_series.rolling(self.rsi_n).std().iloc[-1]
        dev = 10.0 if pd.isna(dev) else float(dev)
        upper = 50 + dev
        lower = 50 - dev
        return upper, lower

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        price_series = df[price_col]
        price = float(price_series.iloc[-1])
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]

        returns = price_series.pct_change().dropna()
        vol = (
            returns.rolling(self.rsi_n).std().iloc[-1]
            if len(returns) >= self.rsi_n
            else 0.0
        )
        if vol * 10000 < self.min_volatility:
            return None

        trend_dir = 0
        if len(df) >= self.trend_ma:
            ma = price_series.rolling(self.trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_pct = (price - ma) / ma * 100
                if diff_pct > self.trend_threshold:
                    trend_dir = 1
                elif diff_pct < -self.trend_threshold:
                    trend_dir = -1
        elif len(df) >= self.trend_rsi_n:
            trsi = rsi(df, self.trend_rsi_n).iloc[-1]
            if trsi > 50 + self.trend_threshold:
                trend_dir = 1
            elif trsi < 50 - self.trend_threshold:
                trend_dir = -1

        upper, lower = self.auto_threshold(rsi_series)
        if trend_dir == 1:
            upper += self.trend_threshold
        elif trend_dir == -1:
            lower -= self.trend_threshold

        if last_rsi > upper:
            strength = min(1.0, (last_rsi - upper) / (100 - upper))
            side = "sell"
        elif last_rsi < lower:
            strength = min(1.0, (lower - last_rsi) / lower)
            side = "buy"
        else:
            return self.finalize_signal(bar, price, None)
        sig = Signal(side, strength)
        return self.finalize_signal(bar, price, sig)


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate mean reversion signals for backtesting."""

    df = data.copy()
    window = params.get("window", 14)
    threshold = params.get("threshold", 0.0)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)

    ma = df["price"].rolling(window).mean()
    df["signal"] = 0
    df.loc[df["price"] < ma - threshold, "signal"] = 1
    df.loc[df["price"] > ma + threshold, "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "fee", "slippage"]]
