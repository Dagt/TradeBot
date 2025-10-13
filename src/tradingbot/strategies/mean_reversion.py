import math
from collections import defaultdict

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics, timeframe_to_minutes
from ..data.features import rsi
from ..filters.liquidity import LiquidityFilterManager
from ..utils.rolling_quantile import RollingQuantileCache

liquidity = LiquidityFilterManager()


def _normalized_strength(raw: float, *, center: float = 0.68) -> float:
    """Map the raw strength into ``[0, 1]`` with a smooth non-linear scale."""

    if not math.isfinite(raw) or raw <= 0:
        return 0.0

    stretch = 1.25
    slope = 1.65

    scaled = math.log1p(raw * stretch)
    adjusted = (scaled - center) * slope

    try:
        logistic = 1.0 / (1.0 + math.exp(-adjusted))
    except OverflowError:
        logistic = 1.0 if adjusted > 0 else 0.0

    baseline = 1.0 / (1.0 + math.exp(center * slope))
    if logistic <= baseline:
        return 0.0

    normalised = (logistic - baseline) / (1.0 - baseline)
    return max(0.0, min(1.0, normalised))


def _best_quote(bar: dict, side: str) -> float | None:
    keys = (
        ("bid", "best_bid", "bid_px", "bid_price")
        if side == "buy"
        else ("ask", "best_ask", "ask_px", "ask_price")
    )
    for key in keys:
        if key not in bar:
            continue
        try:
            value = float(bar[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _pivot_price(df: pd.DataFrame, side: str, lookback: int = 5) -> float | None:
    if side == "buy":
        series = df["low"] if "low" in df else df["close"]
        value = series.iloc[-lookback:].min()
    else:
        series = df["high"] if "high" in df else df["close"]
        value = series.iloc[-lookback:].max()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None

PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "trend_ma": "Ventana para la media móvil de tendencia",
    "trend_rsi_n": "Ventana del RSI para medir tendencia",
    "trend_ma_bps": "Desviación mínima frente a la MA (en puntos básicos)",
    "trend_rsi_shift": "Desplazamiento base del RSI para filtrar tendencias",
    "trend_rsi_shift_max": "Límite máximo del desplazamiento dinámico del RSI",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class MeanReversion(Strategy):
    """RSI based mean reversion strategy with adaptive strength.

    Generates ``buy`` or ``sell`` signals when the RSI deviates from
    dynamically determined thresholds. Signal strength scales with the
    distance from the threshold. The regime filter now distinguishes between
    price extensions over the trend moving average (``trend_ma_bps`` expressed
    in basis points) and RSI displacement (``trend_rsi_shift``) that expands
    with recent volatility up to ``trend_rsi_shift_max``.
    """

    name = "mean_reversion"
    max_signal_strength = 2.5

    TIMEFRAME_OVERRIDES: dict[str, dict[str, float | int | bool]] = {
        "3m": {
            "rsi_n": 17,
            "trend_ma": 85,
            "trend_rsi_n": 75,
            "trend_ma_bps": 235.0,
            "trend_rsi_shift": 7.5,
            "trend_rsi_shift_max": 22.5,
            "min_volatility": 0.52,
            "vol_floor_quantile": 0.24,
            "vol_floor_window": 160,
            "vol_floor_min_periods": 40,
            "strength_gain": 3.6,
            "rsi_dev_floor": 9.0,
            "rsi_dev_cap": 26.0,
            "limit_span_multiplier": 1.16,
            "target_distance_multiplier": 1.30,
            "cooldown_bars": 2,
            "time_stop": 15,
            "only_buy_dip": False,
            "chase_quotes": True,
            "maker_patience": 3,
            "step_mult": 0.38,
            "min_strength": 0.18,
            "span_vol_scaler": 0.55,
            "target_vol_scaler": 0.45,
            "min_strength_low_vol_mult": 1.5,
            "min_strength_high_vol_mult": 0.55,
        },
        "5m": {
            "rsi_n": 15,
            "trend_ma": 75,
            "trend_rsi_n": 65,
            "trend_ma_bps": 205.0,
            "trend_rsi_shift": 6.0,
            "trend_rsi_shift_max": 18.0,
            "min_volatility": 0.42,
            "vol_floor_quantile": 0.20,
            "vol_floor_window": 120,
            "vol_floor_min_periods": 30,
            "strength_gain": 3.4,
            "rsi_dev_floor": 8.5,
            "rsi_dev_cap": 23.0,
            "limit_span_multiplier": 1.08,
            "target_distance_multiplier": 1.22,
            "cooldown_bars": 2,
            "time_stop": 12,
            "only_buy_dip": False,
            "chase_quotes": True,
            "maker_patience": 3,
            "step_mult": 0.34,
            "min_strength": 0.15,
            "span_vol_scaler": 0.50,
            "target_vol_scaler": 0.40,
            "min_strength_low_vol_mult": 1.4,
            "min_strength_high_vol_mult": 0.6,
        },
        "15m": {
            "trend_ma": 65,
            "trend_rsi_n": 55,
            "trend_ma_bps": 168.0,
            "trend_rsi_shift": 5.0,
            "trend_rsi_shift_max": 16.0,
            "min_volatility": 0.36,
            "vol_floor_quantile": 0.20,
            "vol_floor_window": 105,
            "vol_floor_min_periods": 26,
            "strength_gain": 3.1,
            "rsi_dev_floor": 5.2,
            "rsi_dev_cap": 17.5,
            "limit_span_multiplier": 1.01,
            "target_distance_multiplier": 1.08,
            "cooldown_bars": 0,
            "time_stop": 11,
            "only_buy_dip": False,
            "chase_quotes": True,
            "maker_patience": 2,
            "step_mult": 0.26,
            "min_strength": 0.0,
            "span_vol_scaler": 0.45,
            "target_vol_scaler": 0.35,
            "min_strength_low_vol_mult": 1.3,
            "min_strength_high_vol_mult": 0.5,
        },
        "30m": {
            "trend_ma": 55,
            "trend_rsi_n": 50,
            "trend_ma_bps": 162.0,
            "trend_rsi_shift": 4.5,
            "trend_rsi_shift_max": 14.0,
            "min_volatility": 0.30,
            "strength_gain": 2.7,
            "rsi_dev_floor": 5.0,
            "rsi_dev_cap": 16.0,
            "limit_span_multiplier": 0.96,
            "target_distance_multiplier": 1.10,
            "cooldown_bars": 1,
            "time_stop": 9,
            "only_buy_dip": True,
            "chase_quotes": True,
            "maker_patience": 2,
            "step_mult": 0.25,
            "min_strength": 0.07,
            "span_vol_scaler": 0.40,
            "target_vol_scaler": 0.32,
            "min_strength_low_vol_mult": 1.25,
            "min_strength_high_vol_mult": 0.55,
        },
        "1h": {
            "trend_ma": 55,
            "trend_rsi_n": 45,
            "trend_ma_bps": 148.0,
            "trend_rsi_shift": 4.2,
            "trend_rsi_shift_max": 12.0,
            "min_volatility": 0.26,
            "strength_gain": 2.5,
            "rsi_dev_floor": 4.5,
            "rsi_dev_cap": 13.5,
            "cooldown_bars": 1,
            "time_stop": 7,
            "only_buy_dip": True,
            "chase_quotes": True,
            "maker_patience": 2,
            "step_mult": 0.24,
            "min_strength": 0.05,
            "span_vol_scaler": 0.35,
            "target_vol_scaler": 0.28,
            "min_strength_low_vol_mult": 1.2,
            "min_strength_high_vol_mult": 0.6,
        },
        "4h": {
            "trend_ma": 48,
            "trend_rsi_n": 40,
            "trend_ma_bps": 120.0,
            "trend_rsi_shift": 3.6,
            "trend_rsi_shift_max": 10.5,
            "min_volatility": 0.24,
            "strength_gain": 2.1,
            "rsi_dev_floor": 3.6,
            "rsi_dev_cap": 12.5,
            "cooldown_bars": 1,
            "time_stop": 6,
            "only_buy_dip": True,
            "chase_quotes": True,
            "maker_patience": 1,
            "step_mult": 0.22,
            "min_strength": 0.04,
            "span_vol_scaler": 0.28,
            "target_vol_scaler": 0.24,
            "min_strength_low_vol_mult": 1.15,
            "min_strength_high_vol_mult": 0.65,
        },
    }

    def __init__(self, **kwargs):
        tf = str(kwargs.get("timeframe", "1m"))
        overrides = self.TIMEFRAME_OVERRIDES.get(tf, {})
        for key, value in overrides.items():
            kwargs.setdefault(key, value)

        self.timeframe = tf
        tf_minutes = timeframe_to_minutes(tf)
        self._base_timeframe_minutes = tf_minutes

        self.strength_gain = float(kwargs.get("strength_gain", 3.0))
        self._base_strength_gain = self.strength_gain
        self._rsi_dev_floor = float(kwargs.get("rsi_dev_floor", 5.0))
        self._rsi_dev_cap = float(kwargs.get("rsi_dev_cap", 20.0))
        self._limit_span_mult = float(kwargs.get("limit_span_multiplier", 1.0))
        self._base_limit_span_mult = self._limit_span_mult
        self._target_distance_mult = float(kwargs.get("target_distance_multiplier", 1.0))
        self._base_target_distance_mult = self._target_distance_mult
        self._step_mult = max(0.0, float(kwargs.get("step_mult", 0.45)))
        self._base_step_mult = self._step_mult
        self._chase_quotes = bool(kwargs.get("chase_quotes", True))
        self._base_chase_quotes = self._chase_quotes
        maker_patience = kwargs.get("maker_patience")
        self._maker_patience_override = (
            max(0, int(maker_patience)) if maker_patience is not None else None
        )
        self._min_strength = max(
            0.0, min(1.0, float(kwargs.get("min_strength", 0.0)))
        )
        self._base_min_strength = self._min_strength
        self._span_vol_scaler = max(
            0.0, float(kwargs.get("span_vol_scaler", 0.35))
        )
        self._target_vol_scaler = max(
            0.0, float(kwargs.get("target_vol_scaler", 0.25))
        )
        self._min_strength_low_vol_mult = max(
            0.1, float(kwargs.get("min_strength_low_vol_mult", 1.2))
        )
        self._min_strength_high_vol_mult = max(
            0.1, float(kwargs.get("min_strength_high_vol_mult", 0.6))
        )
        self._cooldown_bars = int(kwargs.get("cooldown_bars", 0))
        self._cooldowns: dict[str, int] = defaultdict(int)

        self.rsi_n = int(kwargs.get("rsi_n", 14))

        trend_ma_min = kwargs.get("trend_ma", 50)
        self.trend_ma = max(1, int(trend_ma_min / tf_minutes))

        trend_rsi_min = kwargs.get("trend_rsi_n", 50)
        self.trend_rsi_n = max(1, int(trend_rsi_min / tf_minutes))

        self.trend_ma_bps = max(0.0, float(kwargs.get("trend_ma_bps", 180.0)))
        base_shift = max(0.0, float(kwargs.get("trend_rsi_shift", 5.0)))
        self.trend_rsi_shift = base_shift
        shift_cap = float(kwargs.get("trend_rsi_shift_max", 18.0))
        self.trend_rsi_shift_max = max(base_shift, shift_cap) if shift_cap > 0 else base_shift

        self.min_volatility = max(0.0, float(kwargs.get("min_volatility", 0.0)))
        self.vol_floor_quantile = float(kwargs.get("vol_floor_quantile", 0.25))
        if self.vol_floor_quantile < 0:
            self.vol_floor_quantile = 0.0
        default_floor_window = max(40, self.rsi_n * 2)
        self._vol_floor_window = max(
            5,
            int(kwargs.get("vol_floor_window", default_floor_window)),
        )
        self._vol_floor_min_periods = max(
            3,
            int(kwargs.get("vol_floor_min_periods", max(5, self.rsi_n))),
        )
        self.only_buy_dip = kwargs.get("only_buy_dip", tf in {"30m", "1h"})
        self._base_only_buy_dip = self.only_buy_dip
        default_time_stop_bars = 0.0
        min_time_stop_bars = 1
        if tf_minutes >= 30.0:
            if tf_minutes < 60.0:
                min_time_stop_bars = 5
            else:
                min_time_stop_bars = 3
            default_time_stop_bars = float(min_time_stop_bars)

        self._time_stop_target_bars = 0
        self._min_time_stop_bars = 0
        time_stop_param = float(kwargs.get("time_stop", default_time_stop_bars))
        if time_stop_param > 0:
            target_bars = int(math.ceil(time_stop_param))
            target_bars = max(min_time_stop_bars, target_bars)
            self._time_stop_target_bars = target_bars
            self._min_time_stop_bars = min_time_stop_bars

        self._time_stop_minutes = self._time_stop_target_bars * tf_minutes if self._time_stop_target_bars else 0
        self.time_stop = 0
        self._open_bars: dict[str, int] = {}
        self.risk_service = kwargs.get("risk_service")
        self._rq = RollingQuantileCache()

    def _vol_floor_windows(self, tf_minutes: float) -> tuple[int, int]:
        base_minutes = self._base_timeframe_minutes or tf_minutes
        if tf_minutes <= 0 or base_minutes <= 0:
            window = max(5, self._vol_floor_window)
            min_periods = max(1, min(window, self._vol_floor_min_periods))
            return window, min_periods

        ratio = tf_minutes / base_minutes
        ratio = max(ratio, 1e-9)
        scaled_window = int(math.ceil(self._vol_floor_window / ratio))
        window = max(5, scaled_window)
        scaled_min = int(math.ceil(self._vol_floor_min_periods / ratio))
        min_periods = max(1, min(window, max(1, scaled_min)))
        return window, min_periods

    def auto_threshold(
        self, rsi_series: pd.Series, *, lookback: int | None = None
    ) -> tuple[float, float]:
        """Derive upper and lower RSI bounds from recent variability."""

        window = lookback or self.rsi_n
        dev = rsi_series.rolling(window).std().iloc[-1]
        dev = 10.0 if pd.isna(dev) else float(dev)
        dev = max(self._rsi_dev_floor, min(self._rsi_dev_cap, dev))
        upper = 50 + dev
        lower = 50 - dev
        return upper, lower

    def _adaptive_rsi_period(self, tf_minutes: float, window_size: int) -> int:
        base_period = max(5, self.rsi_n)
        if window_size <= base_period + 1:
            return base_period

        if tf_minutes <= 3.0:
            factor = 0.7
        elif tf_minutes <= 5.0:
            factor = 0.78
        elif tf_minutes <= 15.0:
            factor = 0.9
        else:
            factor = 1.0

        period = int(round(base_period * factor))
        period = max(5, min(window_size - 1, period))
        return period

    def _detect_market_state(
        self,
        price_series: pd.Series,
        returns: pd.Series,
        rsi_series: pd.Series,
        trend_dir: int,
        vol_ratio: float,
        tf_minutes: float,
    ) -> tuple[str, dict[str, float]]:
        lookback = min(len(price_series), max(12, int(round(90 / max(tf_minutes, 1.0)))))
        if lookback < 6:
            return "range", {"strength": 0.0}

        window = price_series.iloc[-lookback:]
        start_price = float(window.iloc[0])
        end_price = float(window.iloc[-1])
        if not math.isfinite(start_price) or not math.isfinite(end_price) or start_price == 0:
            return "range", {"strength": 0.0}

        drift = (end_price - start_price) / start_price
        swing = (window.max() - window.min()) / start_price
        swing = max(swing, 1e-6)
        momentum = returns.iloc[-lookback:].mean() if len(returns) >= lookback else 0.0
        rsi_tail = rsi_series.iloc[-min(len(rsi_series), lookback):]
        rsi_drift = float(rsi_tail.diff().mean()) if len(rsi_tail) >= 2 else 0.0

        tf_scale = math.sqrt(max(tf_minutes, 1.0) / 5.0)
        trend_threshold = 0.0025 * tf_scale
        breakout_threshold = 0.004 * tf_scale
        drift_abs = abs(drift)
        momentum_abs = abs(momentum)

        regime_strength = drift_abs / swing
        metrics = {
            "strength": regime_strength,
            "drift": drift,
            "momentum": momentum,
            "rsi_drift": rsi_drift,
        }

        if vol_ratio >= 1.45 and drift_abs >= breakout_threshold:
            return "breakout", metrics
        if drift_abs >= trend_threshold or (trend_dir != 0 and regime_strength > 0.35):
            return "trend", metrics
        if vol_ratio <= 0.9 and swing < 0.002 * tf_scale:
            return "quiet", metrics
        if momentum_abs <= trend_threshold * 0.3 and abs(rsi_drift) <= 0.4:
            return "range", metrics
        if vol_ratio >= 1.25:
            return "volatile", metrics
        return "range", metrics

    def _dynamic_calibration(
        self,
        *,
        tf_minutes: float,
        market_state: str,
        trend_dir: int,
        vol_ratio: float,
        regime_strength: float,
    ) -> dict[str, float | bool | int]:
        strength_gain = self._base_strength_gain
        limit_span_mult = self._base_limit_span_mult
        target_distance_mult = self._base_target_distance_mult
        step_mult = self._base_step_mult
        min_strength = self._base_min_strength
        chase_quotes = self._base_chase_quotes
        only_buy_dip = self._base_only_buy_dip
        cooldown = self._cooldown_bars
        maker_bias = 0

        if tf_minutes <= 5.0:
            limit_span_mult *= 0.82
            target_distance_mult *= 0.78
            step_mult *= 0.92
            min_strength *= 0.7
            strength_gain *= 1.08
            cooldown = max(0, int(round(cooldown * 0.6)))
            chase_quotes = True
        elif tf_minutes <= 15.0:
            limit_span_mult *= 0.9
            target_distance_mult *= 0.88
            min_strength *= 0.85
            strength_gain *= 1.03
            cooldown = max(0, int(round(cooldown * 0.8)))
        elif tf_minutes >= 60.0:
            limit_span_mult *= 1.12
            target_distance_mult *= 1.1
            step_mult *= 1.05
            min_strength *= 1.05

        if market_state == "trend":
            bias = 1.0 + min(0.6, regime_strength)
            limit_span_mult *= 1.05 * bias
            target_distance_mult *= 1.12 * bias
            min_strength *= 1.08
            strength_gain *= 0.95
            only_buy_dip = trend_dir >= 0 or only_buy_dip
            maker_bias -= 1
        elif market_state == "breakout":
            limit_span_mult *= 0.75
            target_distance_mult *= 0.9
            min_strength *= 0.6
            strength_gain *= 1.2
            step_mult *= 1.15
            chase_quotes = True
            maker_bias -= 1
        elif market_state == "volatile":
            limit_span_mult *= 1.02
            target_distance_mult *= 1.05
            step_mult *= 1.12
            min_strength *= 0.92
            maker_bias -= 1
        elif market_state == "quiet":
            limit_span_mult *= 0.85
            target_distance_mult *= 0.8
            min_strength *= 0.75
            strength_gain *= 1.1
            step_mult *= 0.95
        else:  # range
            limit_span_mult *= 0.88
            target_distance_mult *= 0.85
            min_strength *= 0.82
            step_mult *= 0.98

        if vol_ratio >= 1.35:
            maker_bias -= 1
            chase_quotes = True
            min_strength *= 0.9
        elif vol_ratio <= 0.8:
            limit_span_mult *= 0.9
            target_distance_mult *= 0.82
            step_mult *= 0.9

        maker_bias = int(max(-2, min(1, maker_bias)))

        rsi_period = self.rsi_n
        if tf_minutes <= 5.0 or market_state in {"breakout", "quiet"}:
            if market_state == "quiet":
                rsi_period = max(5, int(round(self.rsi_n * 0.65)))
            else:
                rsi_period = max(5, int(round(self.rsi_n * 0.75)))
        elif market_state == "trend":
            rsi_period = max(5, int(round(self.rsi_n * 0.9)))

        return {
            "strength_gain": float(max(0.1, strength_gain)),
            "limit_span_mult": float(max(0.1, limit_span_mult)),
            "target_distance_mult": float(max(0.1, target_distance_mult)),
            "step_mult": float(max(0.0, step_mult)),
            "min_strength": float(max(0.0, min(1.0, min_strength))),
            "chase_quotes": bool(chase_quotes),
            "only_buy_dip": bool(only_buy_dip),
            "cooldown_bars": int(max(0, cooldown)),
            "maker_patience_bias": maker_bias,
            "rsi_period": int(max(5, rsi_period)),
        }

    def _trend_rsi_offset(self, recent_vol_bps: float) -> float:
        base = self.trend_rsi_shift
        if base <= 0:
            return 0.0
        vol_component = math.log1p(max(recent_vol_bps, 0.0) / 50.0)
        offset = base * (1.0 + vol_component)
        if self.trend_rsi_shift_max > 0:
            return min(offset, self.trend_rsi_shift_max)
        return offset

    def _confirm_reversal(
        self,
        price_series: pd.Series,
        rsi_series: pd.Series,
        side: str,
    ) -> bool:
        """Relaxed reversal confirmation for 5m/15m bars.

        Historically the strategy required the immediately previous tick to
        point in the opposite direction before acting on an RSI excursion.  On
        medium intraday bars (5m/15m) that filter was too strict and often
        missed fills even when short term momentum had already stalled.

        We now confirm the reversal whenever a short moving average or a small
        tolerance band indicates exhaustion, giving a slightly wider window for
        execution while still blocking momentum trades.
        """

        if self.timeframe not in {"5m", "15m"}:
            return True
        if side not in {"buy", "sell"}:
            return True
        if len(price_series) < 2 or len(rsi_series) < 2:
            return False

        last_price = float(price_series.iloc[-1])
        prev_price = float(price_series.iloc[-2])
        last_rsi = float(rsi_series.iloc[-1])
        prev_rsi = float(rsi_series.iloc[-2])

        price_tol = 0.001 if self.timeframe == "5m" else 0.0015
        rsi_tol = 0.7 if self.timeframe == "5m" else 0.9

        ma_window = 3 if self.timeframe == "5m" else 4
        price_ma = price_series.rolling(ma_window, min_periods=1).mean()
        rsi_ma = rsi_series.rolling(ma_window, min_periods=1).mean()

        price_ma_curr = float(price_ma.iloc[-1])
        price_ma_prev = float(price_ma.iloc[-2]) if len(price_ma) >= 2 else prev_price
        rsi_ma_curr = float(rsi_ma.iloc[-1])
        rsi_ma_prev = float(rsi_ma.iloc[-2]) if len(rsi_ma) >= 2 else prev_rsi

        price_change = 0.0
        if prev_price:
            price_change = (last_price - prev_price) / prev_price

        price_ma_slope = price_ma_curr - price_ma_prev
        rsi_ma_slope = rsi_ma_curr - rsi_ma_prev

        if side == "sell":
            return (
                price_change <= price_tol
                or price_ma_slope <= last_price * price_tol
                or last_rsi <= prev_rsi + rsi_tol
                or rsi_ma_slope <= rsi_tol
            )

        # side == "buy"
        return (
            price_change >= -price_tol
            or price_ma_slope >= -last_price * price_tol
            or last_rsi >= prev_rsi - rsi_tol
            or rsi_ma_slope >= -rsi_tol
        )

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        symbol = str(bar.get("symbol", "") or "")
        min_required = max(self.rsi_n, 6)
        if len(df) < min_required + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        price_series = df[price_col]
        price = float(price_series.iloc[-1])
        tf_minutes = timeframe_to_minutes(bar.get("timeframe", self.timeframe))
        rsi_period = self._adaptive_rsi_period(tf_minutes, len(df))
        rsi_series = rsi(df, rsi_period)
        last_rsi = rsi_series.iloc[-1]

        bar_timeframe = bar.get("timeframe", self.timeframe)
        tf_minutes = timeframe_to_minutes(bar_timeframe)
        base_minutes = self._base_timeframe_minutes
        if self._time_stop_target_bars <= 0 or tf_minutes <= 0:
            time_stop_bars = 0
        else:
            desired_minutes = self._time_stop_target_bars * base_minutes
            scaled_bars = int(math.ceil(desired_minutes / tf_minutes))
            min_bars = self._min_time_stop_bars or 1
            time_stop_bars = max(min_bars, scaled_bars)
        self.time_stop = time_stop_bars

        if time_stop_bars and self.risk_service is not None and bar.get("symbol"):
            sym = bar["symbol"]
            trade = self.risk_service.get_trade(sym)
            if trade:
                cnt = self._open_bars.get(sym, 0) + 1
                self._open_bars[sym] = cnt
                if cnt >= time_stop_bars:
                    side_exit = "sell" if trade.get("side") == "buy" else "buy"
                    return self.finalize_signal(bar, price, Signal(side_exit, 1.0))
            else:
                self._open_bars[sym] = 0

        returns = price_series.pct_change().dropna()
        vol_series = (
            returns.rolling(self.rsi_n).std()
            if len(returns) >= self.rsi_n
            else pd.Series(dtype=float)
        )
        vol = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
        vol_bps = vol * 10000 if math.isfinite(vol) and vol > 0 else 0.0

        vol_floor_bps = self.min_volatility
        symbol = str(bar.get("symbol", "") or "")
        if (
            symbol
            and self.vol_floor_quantile > 0
            and math.isfinite(vol_bps)
        ):
            window, min_periods = self._vol_floor_windows(tf_minutes)
            rq = self._rq.get(
                symbol,
                "volatility_floor",
                window=window,
                q=self.vol_floor_quantile,
                min_periods=min_periods,
            )
            floor_candidate = float(rq.update(float(vol_bps)))
            if math.isfinite(floor_candidate) and floor_candidate > 0:
                vol_floor_bps = max(vol_floor_bps, floor_candidate)

        scaled_floor = vol_floor_bps
        base_minutes = self._base_timeframe_minutes or tf_minutes
        if vol_floor_bps > 0 and tf_minutes > 0 and base_minutes > 0:
            scaled_floor = vol_floor_bps * math.sqrt(tf_minutes / base_minutes)
        vol_ratio = 1.0
        if scaled_floor > 1e-9:
            vol_ratio = vol_bps / max(scaled_floor, 1e-9)
        vol_ratio = max(0.2, min(3.5, vol_ratio))
        if scaled_floor <= 1e-9 and self.vol_floor_quantile > 0 and vol_bps <= 0:
            return None
        if scaled_floor > 0 and vol_bps < scaled_floor:
            return None
        abs_price = max(abs(price), 1e-9)
        price_vol = abs_price * vol if math.isfinite(vol) and vol > 0 else 0.0
        bar["volatility"] = price_vol
        target_vol = price_vol
        if len(vol_series.dropna()):
            target_candidate = float(vol_series.dropna().median())
            if math.isfinite(target_candidate) and target_candidate > 0:
                target_vol = abs_price * max(target_candidate, vol)
        bar["target_volatility"] = target_vol

        high = df.get("high")
        low = df.get("low")
        atr_val = 0.0
        if high is not None and low is not None:
            prev_close = price_series.shift()
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_series = tr.rolling(self.rsi_n).mean().dropna()
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

        trend_dir = 0
        trend_offset = self._trend_rsi_offset(vol_bps)
        if len(df) >= self.trend_ma:
            ma = price_series.rolling(self.trend_ma).mean().iloc[-1]
            if not pd.isna(ma) and ma != 0:
                diff_bps = (price - ma) / ma * 10000
                if diff_bps > self.trend_ma_bps:
                    trend_dir = 1
                elif diff_bps < -self.trend_ma_bps:
                    trend_dir = -1
        elif len(df) >= self.trend_rsi_n:
            trsi = rsi(df, self.trend_rsi_n).iloc[-1]
            if trsi > 50 + trend_offset:
                trend_dir = 1
            elif trsi < 50 - trend_offset:
                trend_dir = -1

        market_state, regime_metrics = self._detect_market_state(
            price_series,
            returns,
            rsi_series,
            trend_dir,
            vol_ratio,
            tf_minutes,
        )
        calibration = self._dynamic_calibration(
            tf_minutes=tf_minutes,
            market_state=market_state,
            trend_dir=trend_dir,
            vol_ratio=vol_ratio,
            regime_strength=float(regime_metrics.get("strength", 0.0)),
        )

        if calibration["rsi_period"] != rsi_period:
            new_period = int(calibration["rsi_period"])
            if len(df) >= new_period + 1:
                rsi_series = rsi(df, new_period)
                last_rsi = rsi_series.iloc[-1]
                rsi_period = new_period

        cooldown_dynamic = max(0, int(calibration["cooldown_bars"]))
        if cooldown_dynamic > 0 and symbol:
            remaining = self._cooldowns.get(symbol, 0)
            if remaining > 0:
                self._cooldowns[symbol] = remaining - 1
                return None

        upper, lower = self.auto_threshold(rsi_series, lookback=rsi_period)
        if trend_dir == 1:
            upper += trend_offset
        elif trend_dir == -1:
            lower -= trend_offset

        strength_gain = float(calibration["strength_gain"])
        raw_strength = 0.0
        if last_rsi > upper:
            if not self._confirm_reversal(price_series, rsi_series, "sell"):
                return self.finalize_signal(bar, price, None)
            deviation = (last_rsi - upper) / max(1.0, 100 - upper)
            raw_strength = max(0.0, deviation * strength_gain)
            side = "sell"
        elif last_rsi < lower:
            if not self._confirm_reversal(price_series, rsi_series, "buy"):
                return self.finalize_signal(bar, price, None)
            deviation = (lower - last_rsi) / max(1.0, lower)
            raw_strength = max(0.0, deviation * strength_gain)
            side = "buy"
        else:
            return self.finalize_signal(bar, price, None)

        only_buy_dip = bool(calibration["only_buy_dip"])
        if side == "sell" and trend_dir == 1 and only_buy_dip:
            return self.finalize_signal(bar, price, None)

        strength = _normalized_strength(raw_strength)
        eff_min_strength = max(0.0, float(calibration["min_strength"]))
        if eff_min_strength > 0.0:
            if vol_ratio >= 1.05:
                high_adj = self._min_strength_high_vol_mult / (vol_ratio ** 0.5)
                eff_min_strength *= max(0.1, min(1.0, high_adj))
            elif vol_ratio < 0.9:
                low_adj = self._min_strength_low_vol_mult * (1.0 + (0.9 - vol_ratio))
                eff_min_strength *= max(0.1, min(3.0, low_adj))
        if strength <= 0.0 or strength < eff_min_strength:
            return self.finalize_signal(bar, price, None)

        sig = Signal(side, strength)

        anchor_price = _best_quote(bar, side)
        if anchor_price is None:
            anchor_price = _pivot_price(df, side, lookback=6)
        if anchor_price is None or anchor_price <= 0:
            anchor_price = price

        try:
            tick_size = float(bar.get("tick_size", 0.0) or 0.0)
        except (TypeError, ValueError):
            tick_size = 0.0
        if not math.isfinite(tick_size) or tick_size <= 0:
            tick_size = 0.0

        abs_price = max(abs(price), 1e-9)
        atr_abs = abs(float(atr_val)) if math.isfinite(atr_val) else 0.0
        anchor_gap = abs(float(anchor_price) - price)

        span_boost = 1.0 + self._span_vol_scaler * (vol_ratio - 1.0)
        span_boost = max(0.5, min(1.9, span_boost))
        span_mult = max(0.2, float(calibration["limit_span_mult"]) * span_boost)
        base_span = max(
            abs_price * 0.0004 * span_mult,
            (atr_abs * 0.65 if atr_abs > 0 else 0.0) * span_mult,
            (tick_size * 4 if tick_size else 0.0) * span_mult,
        )
        limit_span = max(base_span, anchor_gap * span_mult)
        if not math.isfinite(limit_span) or limit_span <= 0:
            limit_span = max(abs_price * 0.0004 * span_mult, tick_size * 2 * span_mult if tick_size else 1e-6)

        if side == "buy":
            base_price = max(0.0, anchor_price - limit_span)
        else:
            base_price = anchor_price + limit_span

        target_boost = 1.0 + self._target_vol_scaler * (vol_ratio - 1.0)
        target_boost = max(0.5, min(1.8, target_boost))
        dist_mult = max(0.2, float(calibration["target_distance_mult"]) * target_boost)
        target_distance = max(
            abs_price * 0.00012 * dist_mult,
            (atr_abs * 0.2 if atr_abs > 0 else 0.0) * dist_mult,
            (tick_size if tick_size else 0.0) * dist_mult,
        )
        if anchor_gap > 0:
            target_distance = max(target_distance, min(anchor_gap * 0.25, limit_span))
        target_distance = min(target_distance, limit_span)

        initial_offset = max(0.0, limit_span - target_distance)

        step_distance = max(
            target_distance * 0.5,
            (atr_abs * 0.12 if atr_abs > 0 else 0.0) * dist_mult,
            abs_price * 0.00008 * dist_mult,
        )
        if tick_size:
            step_distance = max(step_distance, tick_size)
        step_distance = min(step_distance, limit_span)
        step_offset = step_distance

        maker_distance = max(
            target_distance - step_distance,
            (tick_size if tick_size else target_distance * 0.5),
        )
        maker_distance = min(max(maker_distance, 0.0), limit_span)
        maker_initial = max(initial_offset, limit_span - maker_distance)
        max_offset = limit_span if limit_span > 0 else initial_offset

        direction = -1.0 if side == "sell" else 1.0
        limit_price = base_price + direction * initial_offset
        if side == "buy":
            limit_price = min(limit_price, anchor_price)
        else:
            limit_price = max(limit_price, anchor_price)
        sig.limit_price = max(0.0, limit_price)
        chase_orders = bool(calibration["chase_quotes"]) or vol_ratio >= 1.05
        if self._maker_patience_override is not None:
            maker_patience = self._maker_patience_override
        else:
            if vol_ratio >= 1.4:
                maker_patience = 1
            elif vol_ratio >= 0.85:
                maker_patience = 2
            else:
                maker_patience = 3
            if tf_minutes <= 5.0:
                maker_patience = max(maker_patience, 2)
        maker_patience = max(0, int(maker_patience + int(calibration["maker_patience_bias"])))
        meta = {
            "base_price": base_price,
            "limit_offset": abs(limit_span),
            "initial_offset": abs(initial_offset),
            "offset_step": abs(step_offset),
            "max_offset": abs(max_offset),
            "step_mult": max(0.0, float(calibration["step_mult"])),
            "chase": chase_orders,
            "maker_initial_offset": abs(maker_initial),
            "maker_patience": maker_patience,
            "post_only": True,
            "market_state": market_state,
        }
        if tick_size:
            meta["tick_size"] = tick_size
        sig.metadata.update(meta)
        partial_tp = {
            "qty_pct": min(0.6, max(0.22, strength * 0.5)),
            "atr_multiple": max(1.2, 0.85 + strength * 0.4),
            "mode": "scale_out",
        }
        max_hold = max(8, int(round(24 / max(tf_minutes, 1.0))))
        sig.metadata['partial_take_profit'] = partial_tp
        sig.metadata['max_hold_bars'] = max_hold
        sig.post_only = True
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(
                strength,
                price,
                volatility=bar.get("volatility"),
                target_volatility=bar.get("target_volatility"),
                clamp=False,
            )
            stop = self.risk_service.initial_stop(price, side, atr_val)
            if (
                side == "sell"
                and trend_dir == 1
                and self.timeframe in {"5m", "15m"}
            ):
                stop = price + 1.5 * atr_val
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "target_volatility": bar.get("target_volatility"),
                "strength": strength,
                "partial_take_profit": partial_tp,
                "max_hold": max_hold,
            }

        result = self.finalize_signal(bar, price, sig)
        cooldown_dynamic = max(0, int(calibration["cooldown_bars"]))
        if result is not None and cooldown_dynamic > 0 and symbol:
            self._cooldowns[symbol] = cooldown_dynamic
        return result


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


