from __future__ import annotations

from dataclasses import dataclass, field

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
    "min_volatility_quantile": "Cuantil usado para estimar el piso de volatilidad",
    "min_volatility_window_mult": "Multiplicador de ventana para estimar la volatilidad mínima",
    "min_volatility_fallbacks": "Piso mínimo de volatilidad por timeframe (en bps)",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_threshold": "Umbral para considerar la tendencia fuerte",
    "trend_penalty_pct": "Penalización porcentual al z-score cuando la tendencia va en contra",
    "trend_extreme_multiplier": "Multiplicador del umbral de tendencia para pausar señales en sesgos extremos",
    "counter_trend_strength_mult": "Factor de reducción del tamaño cuando la tendencia va en contra",
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
        Factor base usado para escalar el tamaño en función de la
        volatilidad reciente (expresada en puntos básicos).  El valor se
        adapta automáticamente al timeframe mediante la relación
        ``base * sqrt(tf_minutes / 5)`` limitada a un rango prudente, lo
        que resulta en factores efectivos ~``0.015`` en 3 m, ``0.02`` en
        5 m, ``0.05`` en 30 m, ``0.07`` en 1 h y ``0.10`` en 4 h.  Valor
        base por defecto ``0.02``.
    min_volatility : float, optional
        Volatilidad mínima reciente en bps requerida para operar. Si es ``0`` se
        estima dinámicamente a partir de cuantiles recientes y un piso por
        timeframe.
    min_volatility_quantile : float, optional
        Cuantil de la volatilidad histórica (desviación estándar de retornos) a
        utilizar como piso, por defecto ``0.25``.
    min_volatility_window_mult : float, optional
        Multiplicador de la ventana base para estimar el cuantil de
        volatilidad mínima, por defecto ``4.0``.
    min_volatility_fallbacks : dict, optional
        Mapa ``{minutos: bps}`` con el piso mínimo de volatilidad por
        timeframe, por defecto ``{1: 0.1, 5: 0.3, 15: 0.5, 30: 0.7, 60: 0.9}``.
    trend_ma : int, optional
        Window for the moving average used to gauge trend, by default ``50``.
    trend_rsi_n : int, optional
        Window for the RSI used to gauge trend when MA is unavailable,
        by default ``50``.
    trend_threshold : float, optional
        Threshold (% over MA or RSI points) to treat the trend as strong,
        by default ``10.0``.
    trend_penalty_pct : float, optional
        Incremento porcentual requerido sobre el ``z_threshold`` cuando la
        tendencia va en contra, por defecto ``75.0``.
    """

    lookback: int = 15
    z_threshold: float = 0.2
    volatility_factor: float = 0.02
    min_volatility: float = 0.0
    min_volatility_quantile: float = 0.25
    min_volatility_window_mult: float = 4.0
    min_volatility_fallbacks: dict[float, float] = field(
        default_factory=lambda: {
            1.0: 0.1,
            5.0: 0.3,
            15.0: 0.5,
            30.0: 0.7,
            60.0: 0.9,
            120.0: 1.2,
            240.0: 1.5,
        }
    )
    trend_ma: int = 50
    trend_rsi_n: int = 50
    trend_threshold: float = 10.0
    trend_penalty_pct: float = 75.0
    trend_extreme_multiplier: float = 2.0
    counter_trend_strength_mult: float = 0.5


liquidity = LiquidityFilterManager()

# Enforce a floor on rolling windows when working with higher timeframe bars so
# that momentum and mean-reversion indicators never collapse to just a handful
# of observations.
MIN_BARS = 5


def _scaled_volatility_factor(tf_minutes: float, base_factor: float) -> float:
    """Scale the base volatility factor according to the bar timeframe."""

    minutes = max(float(tf_minutes), 1.0)
    base = max(float(base_factor), 0.0)
    scaled = base * math.sqrt(minutes / 5.0)
    return min(max(scaled, 0.01), 0.10)


def _vol_size_bounds(tf_minutes: float) -> tuple[float, float]:
    """Return timeframe-aware clamps for the volatility sized strength."""

    minutes = max(float(tf_minutes), 1.0)
    if minutes <= 3.0:
        return 0.1, 1.2
    if minutes <= 5.0:
        return 0.15, 1.5
    if minutes <= 15.0:
        return 0.2, 2.0
    if minutes <= 30.0:
        return 0.25, 2.5
    if minutes <= 60.0:
        return 0.3, 3.2
    if minutes <= 120.0:
        return 0.35, 3.6
    if minutes <= 240.0:
        return 0.4, 4.0
    return 0.45, 4.5


class ScalpPingPong(Strategy):
    """Mean-reversion scalping strategy using z-score of returns."""

    name = "scalp_pingpong"
    # Keep the clamp for ``raw_size`` and the normalisation ceiling in sync so that the
    # strategy can use the entire [0, 1] strength range when volatility sizing saturates.
    max_signal_strength = 3.0

    def __init__(
        self,
        cfg: ScalpPingPongConfig | None = None,
        *,
        config_path: str | None = None,
        **kwargs,
    ):
        params = {**load_params(config_path), **kwargs}
        risk_service = params.pop("risk_service", None)
        min_strength_param = params.pop("min_strength_fraction", None)
        super().__init__(min_strength_fraction=min_strength_param)
        tf = str(params.pop("timeframe", "1m"))
        self.cfg = cfg or ScalpPingPongConfig(**params)
        self.risk_service = risk_service
        self.timeframe = tf
        self._last_vol_floor_bps = 0.0

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

    def _fallback_vol_floor(self, tf_minutes: float) -> float:
        mapping = getattr(self.cfg, "min_volatility_fallbacks", {}) or {}
        if not mapping:
            return 0.1
        try:
            items = sorted((float(k), float(v)) for k, v in mapping.items())
        except (TypeError, ValueError):
            return 0.1
        floor = items[0][1]
        for minute_mark, value in items:
            floor = value
            if tf_minutes <= minute_mark:
                return floor
        return floor

    def _dynamic_vol_floor(
        self,
        returns: pd.Series,
        lookback: int,
        tf_minutes: float,
    ) -> float:
        quantile = float(getattr(self.cfg, "min_volatility_quantile", 0.25))
        quantile = min(max(quantile, 0.0), 1.0)
        window_mult = max(float(getattr(self.cfg, "min_volatility_window_mult", 4.0)), 1.0)
        base_window = max(MIN_BARS, int(lookback))
        vol_series = returns.rolling(base_window).std().dropna()
        dynamic = 0.0
        latest_bps = 0.0
        if not vol_series.empty:
            latest_bps = float(vol_series.iloc[-1]) * 10000.0
        if not vol_series.empty:
            hist_window = int(max(base_window, int(math.ceil(base_window * window_mult))))
            hist_window = min(len(vol_series), hist_window)
            recent = vol_series.iloc[-hist_window:]
            quant = float(recent.quantile(quantile))
            if math.isfinite(quant) and quant > 0:
                dynamic = quant * 10000.0
                dynamic *= 1.2
        if not math.isfinite(dynamic) or dynamic <= 0:
            dynamic = 0.0
        if latest_bps > 0 and dynamic <= 0:
            dynamic = latest_bps
        if not math.isfinite(dynamic) or dynamic <= 0:
            return 0.0
        return dynamic

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
        tf_floor = float(getattr(self.cfg, "min_volatility", 0.0))
        fallback_floor = self._fallback_vol_floor(tf_minutes)
        if tf_floor <= 0:
            vol_floor = self._dynamic_vol_floor(returns, lookback, tf_minutes)
        else:
            vol_floor = tf_floor
        vol_floor = max(vol_floor, fallback_floor)
        if vol_bps > 0 and bar.get("atr") is not None:
            cap_floor = vol_bps
            if vol_floor > cap_floor:
                vol_floor = max(fallback_floor, cap_floor)
        self._last_vol_floor_bps = vol_floor
        bar["vol_floor_bps"] = vol_floor
        if vol_bps < vol_floor * 0.9:
            return None
        abs_price = max(abs(price), 1e-9)
        price_vol = abs_price * (vol_bps / 10000.0)
        bar["volatility"] = price_vol
        target_bps = max(vol_bps, vol_floor)
        bar["target_volatility"] = abs_price * (target_bps / 10000.0)
        scaled_factor = _scaled_volatility_factor(tf_minutes, self.cfg.volatility_factor)
        bar["volatility_factor"] = scaled_factor
        vol_size = vol_bps * scaled_factor
        clamp_min, clamp_max = _vol_size_bounds(tf_minutes)
        vol_size = max(clamp_min, min(clamp_max, vol_size))

        trend_dir = 0
        trend_threshold = max(0.0, float(self.cfg.trend_threshold))
        trend_extreme_mult = max(
            1.0, float(getattr(self.cfg, "trend_extreme_multiplier", 3.0))
        )
        counter_trend_mult = float(
            getattr(self.cfg, "counter_trend_strength_mult", 0.5)
        )
        if not math.isfinite(counter_trend_mult):
            counter_trend_mult = 0.5
        counter_trend_mult = max(0.0, min(1.0, counter_trend_mult))
        if len(closes) >= trend_ma:
            ma = closes.rolling(trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_pct = (price - ma) / ma * 100
                extreme_threshold = trend_threshold * trend_extreme_mult
                if trend_threshold > 0 and trend_extreme_mult > 1.0:
                    if abs(diff_pct) >= extreme_threshold:
                        return None
                if diff_pct > trend_threshold:
                    trend_dir = 1
                elif diff_pct < -trend_threshold:
                    trend_dir = -1
        elif len(closes) >= trend_rsi_n:
            trsi = rsi(df, trend_rsi_n).iloc[-1]
            upper = 50 + trend_threshold
            lower = 50 - trend_threshold
            extreme_threshold = trend_threshold * trend_extreme_mult
            if trend_threshold > 0 and trend_extreme_mult > 1.0:
                extreme_upper = 50 + extreme_threshold
                extreme_lower = 50 - extreme_threshold
                if trsi >= extreme_upper or trsi <= extreme_lower:
                    return None
            if trsi > upper:
                trend_dir = 1
            elif trsi < lower:
                trend_dir = -1

        penalty_mult = 1.0 + max(0.0, float(self.cfg.trend_penalty_pct)) / 100.0
        z_buy = float(self.cfg.z_threshold) * (penalty_mult if trend_dir == -1 else 1.0)
        z_sell = float(self.cfg.z_threshold) * (penalty_mult if trend_dir == 1 else 1.0)
        z_buy = max(z_buy, 1e-9)
        z_sell = max(z_sell, 1e-9)

        if z <= -z_buy:
            side = "buy"
            strength = abs(z) / z_buy
        elif z >= z_sell:
            side = "sell"
            strength = abs(z) / z_sell
        else:
            return None
        strength = max(0.05, min(2.5, strength))
        if (side == "buy" and trend_dir == -1) or (side == "sell" and trend_dir == 1):
            strength *= counter_trend_mult
            strength = max(0.05, min(2.5, strength))
        raw_size = max(0.0, min(3.0, strength * vol_size))
        if raw_size <= 0:
            return self.finalize_signal(bar, price, None)
        if self.max_signal_strength <= 0:
            return self.finalize_signal(bar, price, None)
        normalized = raw_size / self.max_signal_strength
        effective_min = self.min_strength_fraction
        if normalized < effective_min:
            return self.finalize_signal(bar, price, None)
        normalized = min(1.0, normalized)
        sig = Signal(side, normalized)
        base_price = price
        best_bid = bar.get("bid")
        best_ask = bar.get("ask")
        try:
            if side == "buy" and best_bid is not None:
                base_price = float(best_bid)
            elif side == "sell" and best_ask is not None:
                base_price = float(best_ask)
        except (TypeError, ValueError):
            base_price = price
        offset = max(price_vol * 0.5, abs_price * 0.0005)
        direction = -1 if side == "buy" else 1
        sig.limit_price = base_price + direction * offset
        max_offset = abs(price_vol * 1.5)
        maker_patience = 2 if tf_minutes <= 2.0 else 1
        partial_tp = {
            "qty_pct": min(0.5, max(0.2, sig.strength * 0.5)),
            "atr_multiple": max(1.15, 0.9 + sig.strength * 0.5),
            "mode": "scale_out",
        }
        max_hold = max(6, int(round(16 / max(tf_minutes, 1.0))))
        sig.post_only = True
        sig.metadata.update(
            {
                "base_price": base_price,
                "limit_offset": abs(offset),
                "offset_step": abs(offset) * 0.6,
                "max_offset": max_offset if max_offset > 0 else abs(offset) * 3,
                "step_mult": 0.4,
                "chase": False,
                "decay": 0.6,
                "min_offset": abs_price * 0.0002,
                "post_only": True,
                "maker_initial_offset": abs(offset),
                "maker_patience": maker_patience,
                "partial_take_profit": partial_tp,
                "max_hold_bars": max_hold,
            }
        )

        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(
                sig.strength,
                price,
                volatility=bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
                clamp=False,
            )
            atr_val = bar.get("atr")
            if atr_val is None:
                atr_val = price_vol
            stop = self.risk_service.initial_stop(price, side, atr_val)
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "target_volatility": bar.get("target_volatility"),
                "bars_held": 0,
                "max_hold": max_hold,
                "strength": sig.strength,
                "partial_take_profit": partial_tp,
            }

        return self.finalize_signal(bar, price, sig)
