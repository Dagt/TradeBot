import pandas as pd
from .base import Strategy, Signal, record_signal_metrics

PARAM_INFO = {
    "lookback": "Ventana para medias y desviación estándar",
    "mult": "Multiplicador aplicado a la desviación",
    "volatility_factor": "Factor para dimensionar según volatilidad",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class BreakoutVol(Strategy):
    """Volatility breakout strategy using rolling standard deviation."""

    name = "breakout_vol"

    def __init__(self, **kwargs):
        self.lookback = kwargs.get("lookback", 10)
        self.mult = kwargs.get("mult", 1.5)
        self.volatility_factor = kwargs.get("volatility_factor", 0.02)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.risk_service = kwargs.get("risk_service")

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.lookback + 1:
            return None
        closes = df["close"]
        mean = closes.rolling(self.lookback).mean().iloc[-1]
        std = closes.rolling(self.lookback).std().iloc[-1]
        last = float(closes.iloc[-1])
        upper = mean + self.mult * std
        lower = mean - self.mult * std

        returns = closes.pct_change().dropna()
        vol = (
            returns.rolling(self.lookback).std().iloc[-1]
            if len(returns) >= self.lookback
            else 0.0
        )
        vol_bps = vol * 10000
        if vol_bps < self.min_volatility:
            return None
        size = max(0.0, min(1.0, vol_bps * self.volatility_factor))

        side: str | None = None
        if last > upper:
            side = "buy"
        elif last < lower:
            side = "sell"
        if side is None:
            return self.finalize_signal(bar, last, None)
        sig = Signal(side, size)
        return self.finalize_signal(bar, last, sig)
