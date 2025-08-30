from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
from ..data.features import rsi


PARAM_INFO = {
    "lookback": "Ventana para el cálculo del z-score",
    "z_threshold": "Z-score absoluto para abrir operación",
    "volatility_factor": "Factor de tamaño según volatilidad",
    "min_volatility": "Volatilidad mínima reciente en bps",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_threshold": "Umbral para considerar la tendencia fuerte",
    "config_path": "Ruta opcional al archivo de configuración",
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
        Multiplier applied to recent volatility (in bps) to size positions,
        default ``0.02``.
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


class ScalpPingPong(Strategy):
    """Mean-reversion scalping strategy using z-score of returns."""

    name = "scalp_pingpong"

    def __init__(
        self,
        cfg: ScalpPingPongConfig | None = None,
        *,
        config_path: str | None = None,
        risk_service=None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        self.cfg = cfg or ScalpPingPongConfig(**params)
        self.risk_service = risk_service
        self.trade: dict | None = None

    def _calc_zscore(self, closes: pd.Series) -> float:
        returns = closes.pct_change().dropna()
        if len(returns) < self.cfg.lookback:
            return 0.0
        mean = returns.rolling(self.cfg.lookback).mean().iloc[-1]
        std = returns.rolling(self.cfg.lookback).std().iloc[-1]
        if std == 0 or pd.isna(std):
            return 0.0
        z = (returns.iloc[-1] - mean) / std
        return float(z)

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.cfg.lookback + 1:
            return None
        closes = df["close"]
        returns = closes.pct_change().dropna()
        z = self._calc_zscore(closes)
        price = float(closes.iloc[-1])
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, price)
            trade_state = {**self.trade, "current_price": price}
            decision = self.risk_service.manage_position(trade_state)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            if decision in {"scale_in", "scale_out"}:
                self.trade["strength"] = trade_state.get("strength", 1.0)
                return Signal(self.trade["side"], self.trade["strength"])
            return None

        vol = (
            returns.rolling(self.cfg.lookback).std().iloc[-1]
            if len(returns) >= self.cfg.lookback
            else 0.0
        )
        vol_bps = vol * 10000
        if vol_bps < self.cfg.min_volatility:
            return None
        vol_size = max(0.0, min(1.0, vol_bps * self.cfg.volatility_factor))

        trend_dir = 0
        if len(closes) >= self.cfg.trend_ma:
            ma = closes.rolling(self.cfg.trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_pct = (price - ma) / ma * 100
                if diff_pct > self.cfg.trend_threshold:
                    trend_dir = 1
                elif diff_pct < -self.cfg.trend_threshold:
                    trend_dir = -1
        elif len(closes) >= self.cfg.trend_rsi_n:
            trsi = rsi(df, self.cfg.trend_rsi_n).iloc[-1]
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
            return None
        if self.risk_service:
            qty = self.risk_service.calc_position_size(size, price)
            trade = {"side": side, "entry_price": price, "qty": qty, "strength": size}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(
                price, side, atr
            )
            trade["atr"] = atr
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        return Signal(side, size)
