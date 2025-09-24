from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import math

from .base import (
    Strategy,
    Signal,
    load_params,
    record_signal_metrics,
    timeframe_to_minutes,
)
from ..data.features import rsi
from ..filters.liquidity import LiquidityFilterManager


PARAM_INFO = {
    "lookback": "Ventana para el cálculo del z-score",
    "z_threshold": "Z-score absoluto para abrir operación",
    "volatility_factor": "Factor de tamaño según volatilidad",
    "min_volatility": "Volatilidad mínima reciente en bps",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_threshold": "Umbral para considerar la tendencia fuerte",
}


@dataclass
class ScalpPingPongConfig:
    """Configuration for :class:`ScalpPingPong`.

    Parameters
    ----------
    lookback : int, optional
        Window length for the z-score calculation, by default ``15``.
    z_threshold : float, optional
        Absolute z-score value required to open a trade, by default ``0.2``.
    volatility_factor : float, optional
        Fraction of the recent volatility (expressed in basis points) used
        to size positions.  For example, con una volatilidad de ``50`` bps
        y un factor de ``0.02`` el aporte de tamaño será ``1.0`` (saturado
        al límite superior).  Valor por defecto ``0.02``.
    min_volatility : float, optional
        Volatilidad mínima reciente en bps requerida para operar, por defecto ``0``.
    trend_ma : int, optional
        Window for the moving average used to gauge trend, by default ``50``.
    trend_rsi_n : int, optional
        Window for the RSI used to gauge trend when MA is unavailable,
        by default ``50``.
    trend_threshold : float, optional
        Threshold (% over MA or RSI points) to treat the trend as strong,
        by default ``10.0``.
    """

    lookback: int = 15
    z_threshold: float = 0.2
    volatility_factor: float = 0.02
    min_volatility: float = 0.0
    trend_ma: int = 50
    trend_rsi_n: int = 50
    trend_threshold: float = 10.0


liquidity = LiquidityFilterManager()

# Enforce a floor on rolling windows when working with higher timeframe bars so
# that momentum and mean-reversion indicators never collapse to just a handful
# of observations.
MIN_BARS = 5


class ScalpPingPong(Strategy):
    """Mean-reversion scalping strategy using z-score of returns."""

    name = "scalp_pingpong"

    def __init__(
        self,
        cfg: ScalpPingPongConfig | None = None,
        *,
        config_path: str | None = None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        params.pop("risk_service", None)
        tf = str(params.pop("timeframe", kwargs.get("timeframe", "1m")))
        self.cfg = cfg or ScalpPingPongConfig(**params)
        self.risk_service = kwargs.get("risk_service")
        self.timeframe = tf

    def _calc_zscore(self, closes: pd.Series, lookback: int) -> float:
        lookback = max(MIN_BARS, int(lookback))
        returns = closes.pct_change().dropna()
        if len(returns) < lookback:
            return 0.0
        window = returns.iloc[-lookback:]
        std = window.std(ddof=1)
        if pd.isna(std) or std <= 0:
            return 0.0
        mean = window.mean()
        z = (window.iloc[-1] - mean) / std
        if pd.isna(z) or not math.isfinite(float(z)):
            return 0.0
        return float(z)

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf_minutes = timeframe_to_minutes(bar.get("timeframe", self.timeframe))
        bar_minutes = max(tf_minutes, 1e-9)
        lookback = max(MIN_BARS, int(math.ceil(self.cfg.lookback / bar_minutes)))
        trend_ma = max(MIN_BARS, int(math.ceil(self.cfg.trend_ma / bar_minutes)))
        trend_rsi_n = max(MIN_BARS, int(math.ceil(self.cfg.trend_rsi_n / bar_minutes)))

        if len(df) < lookback + 1:
            return None
        closes = df["close"]
        returns = closes.pct_change().dropna()
        z = self._calc_zscore(closes, lookback)
        price = float(closes.iloc[-1])

        if bar.get("atr") is not None and price != 0:
            vol_bps = float(bar["atr"]) / abs(price) * 10000
        else:
            vol = (
                returns.rolling(lookback).std().iloc[-1]
                if len(returns) >= lookback
                else 0.0
            )
            vol_bps = vol * 10000
        if vol_bps < self.cfg.min_volatility:
            return None
        abs_price = max(abs(price), 1e-9)
        price_vol = abs_price * (vol_bps / 10000.0)
        bar["volatility"] = price_vol
        target_bps = max(vol_bps, self.cfg.min_volatility)
        bar["target_volatility"] = abs_price * (target_bps / 10000.0)
        # ``vol_bps`` ya está expresada en puntos básicos, por lo que el factor
        # de volatilidad actúa directamente como un multiplicador fraccional.
        # El tamaño final se acota en ``[0, 1]`` para evitar sobre-apalancamiento
        # ante lecturas extremas de volatilidad.
        vol_size = vol_bps * self.cfg.volatility_factor
        vol_size = max(0.0, min(1.0, vol_size))

        trend_dir = 0
        if len(closes) >= trend_ma:
            ma = closes.rolling(trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_pct = (price - ma) / ma * 100
                if diff_pct > self.cfg.trend_threshold:
                    trend_dir = 1
                elif diff_pct < -self.cfg.trend_threshold:
                    trend_dir = -1
        elif len(closes) >= trend_rsi_n:
            trsi = rsi(df, trend_rsi_n).iloc[-1]
            if trsi > 50 + self.cfg.trend_threshold:
                trend_dir = 1
            elif trsi < 50 - self.cfg.trend_threshold:
                trend_dir = -1

        z_buy = self.cfg.z_threshold + (
            self.cfg.trend_threshold / 100 if trend_dir == -1 else 0
        )
        z_sell = self.cfg.z_threshold + (
            self.cfg.trend_threshold / 100 if trend_dir == 1 else 0
        )

        if z <= -z_buy:
            side = "buy"
            strength = min(1.0, abs(z) / z_buy)
        elif z >= z_sell:
            side = "sell"
            strength = min(1.0, abs(z) / z_sell)
        else:
            return None
        size = min(1.0, strength * vol_size)
        if size <= 0:
            return self.finalize_signal(bar, price, None)
        sig = Signal(side, size)
        return self.finalize_signal(bar, price, sig)
