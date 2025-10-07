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
            "rsi_n": 18,
            "trend_ma": 90,
            "trend_rsi_n": 80,
            "trend_ma_bps": 260.0,
            "trend_rsi_shift": 8.0,
            "trend_rsi_shift_max": 24.0,
            "min_volatility": 0.8,
            "vol_floor_quantile": 0.35,
            "vol_floor_window": 180,
            "vol_floor_min_periods": 45,
            "strength_gain": 3.8,
            "rsi_dev_floor": 8.0,
            "rsi_dev_cap": 24.0,
            "limit_span_multiplier": 1.15,
            "target_distance_multiplier": 1.20,
            "cooldown_bars": 5,
            "time_stop": 18,
            "only_buy_dip": False,
        },
        "5m": {
            "rsi_n": 16,
            "trend_ma": 80,
            "trend_rsi_n": 70,
            "trend_ma_bps": 220.0,
            "trend_rsi_shift": 6.5,
            "trend_rsi_shift_max": 20.0,
            "min_volatility": 0.6,
            "vol_floor_quantile": 0.30,
            "vol_floor_window": 150,
            "vol_floor_min_periods": 36,
            "strength_gain": 3.6,
            "rsi_dev_floor": 7.0,
            "rsi_dev_cap": 22.0,
            "limit_span_multiplier": 1.10,
            "target_distance_multiplier": 1.15,
            "cooldown_bars": 4,
            "time_stop": 14,
            "only_buy_dip": False,
        },
        "15m": {
            "trend_ma": 70,
            "trend_rsi_n": 60,
            "trend_ma_bps": 190.0,
            "trend_rsi_shift": 5.5,
            "trend_rsi_shift_max": 18.0,
            "min_volatility": 0.45,
            "vol_floor_quantile": 0.28,
            "vol_floor_window": 120,
            "vol_floor_min_periods": 30,
            "strength_gain": 3.2,
            "rsi_dev_floor": 6.0,
            "rsi_dev_cap": 20.0,
            "limit_span_multiplier": 1.05,
            "target_distance_multiplier": 1.10,
            "cooldown_bars": 3,
            "time_stop": 12,
            "only_buy_dip": False,
        },
        "30m": {
            "trend_ma": 60,
            "trend_rsi_n": 55,
            "trend_ma_bps": 180.0,
            "trend_rsi_shift": 5.0,
            "trend_rsi_shift_max": 16.0,
            "min_volatility": 0.40,
            "strength_gain": 2.8,
            "rsi_dev_floor": 5.5,
            "rsi_dev_cap": 18.0,
            "limit_span_multiplier": 1.00,
            "target_distance_multiplier": 1.05,
            "cooldown_bars": 2,
            "time_stop": 10,
            "only_buy_dip": True,
        },
        "1h": {
            "trend_ma": 60,
            "trend_rsi_n": 50,
            "trend_ma_bps": 160.0,
            "trend_rsi_shift": 4.5,
            "trend_rsi_shift_max": 14.0,
            "min_volatility": 0.35,
            "strength_gain": 2.6,
            "rsi_dev_floor": 5.0,
            "rsi_dev_cap": 16.0,
            "cooldown_bars": 2,
            "time_stop": 8,
            "only_buy_dip": True,
        },
        "4h": {
            "trend_ma": 55,
            "trend_rsi_n": 45,
            "trend_ma_bps": 140.0,
            "trend_rsi_shift": 4.0,
            "trend_rsi_shift_max": 12.0,
            "min_volatility": 0.30,
            "strength_gain": 2.2,
            "rsi_dev_floor": 4.0,
            "rsi_dev_cap": 14.0,
            "cooldown_bars": 1,
            "time_stop": 6,
            "only_buy_dip": True,
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
        self._rsi_dev_floor = float(kwargs.get("rsi_dev_floor", 5.0))
        self._rsi_dev_cap = float(kwargs.get("rsi_dev_cap", 20.0))
        self._limit_span_mult = float(kwargs.get("limit_span_multiplier", 1.0))
        self._target_distance_mult = float(kwargs.get("target_distance_multiplier", 1.0))
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

    def auto_threshold(self, rsi_series: pd.Series) -> tuple[float, float]:
        """Derive upper and lower RSI bounds from recent variability."""

        dev = rsi_series.rolling(self.rsi_n).std().iloc[-1]
        dev = 10.0 if pd.isna(dev) else float(dev)
        dev = max(self._rsi_dev_floor, min(self._rsi_dev_cap, dev))
        upper = 50 + dev
        lower = 50 - dev
        return upper, lower

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
        cooldown = self._cooldown_bars
        if cooldown > 0 and symbol:
            remaining = self._cooldowns.get(symbol, 0)
            if remaining > 0:
                self._cooldowns[symbol] = remaining - 1
                return None
        if len(df) < self.rsi_n + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        price_series = df[price_col]
        price = float(price_series.iloc[-1])
        rsi_series = rsi(df, self.rsi_n)
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

        upper, lower = self.auto_threshold(rsi_series)
        if trend_dir == 1:
            upper += trend_offset
        elif trend_dir == -1:
            lower -= trend_offset

        raw_strength = 0.0
        if last_rsi > upper:
            if not self._confirm_reversal(price_series, rsi_series, "sell"):
                return self.finalize_signal(bar, price, None)
            deviation = (last_rsi - upper) / max(1.0, 100 - upper)
            raw_strength = max(0.0, deviation * self.strength_gain)
            side = "sell"
        elif last_rsi < lower:
            if not self._confirm_reversal(price_series, rsi_series, "buy"):
                return self.finalize_signal(bar, price, None)
            deviation = (lower - last_rsi) / max(1.0, lower)
            raw_strength = max(0.0, deviation * self.strength_gain)
            side = "buy"
        else:
            return self.finalize_signal(bar, price, None)

        if side == "sell" and trend_dir == 1 and self.only_buy_dip:
            return self.finalize_signal(bar, price, None)

        strength = _normalized_strength(raw_strength)
        if strength <= 0.0:
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

        span_mult = max(0.2, self._limit_span_mult)
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

        dist_mult = max(0.2, self._target_distance_mult)
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
        meta = {
            "base_price": base_price,
            "limit_offset": abs(limit_span),
            "initial_offset": abs(initial_offset),
            "offset_step": abs(step_offset),
            "max_offset": abs(max_offset),
            "step_mult": 0.3,
            "chase": True,
            "maker_initial_offset": abs(maker_initial),
            "maker_patience": 1,
            "post_only": True,
        }
        if tick_size:
            meta["tick_size"] = tick_size
        sig.metadata.update(meta)
        sig.metadata.update(
            {
                "base_price": base_price,
                "limit_offset": abs(limit_span),
                "initial_offset": abs(initial_offset),
                "offset_step": abs(step_offset),
                "max_offset": abs(limit_span),
                "step_mult": 0.45,
                "chase": True,
                "maker_initial_offset": abs(maker_initial),
                "maker_patience": 1,
                "post_only": True,
            }
        )
        maker_patience = 2 if tf_minutes <= 5.0 else 1
        sig.metadata['maker_patience'] = maker_patience
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
        if result is not None and cooldown > 0 and symbol:
            self._cooldowns[symbol] = cooldown
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


