import math
from typing import Iterable

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics, timeframe_to_minutes
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager

PARAM_INFO = {
    "lookback": "Ventana para medias y desviación estándar",
    "volatility_factor": "Factor para dimensionar según volatilidad",
    "max_offset_pct": "Porcentaje máximo del precio para el offset del límite",
}


liquidity = LiquidityFilterManager()


def _safe_first_float(container: dict, keys: Iterable[str]) -> float | None:
    """Return the first finite float found in ``container`` for ``keys``.

    The helper inspects both the bar payload and any nested ``"book"`` entry
    to capture best bid/ask prices regardless of the upstream data source.
    """

    book = container.get("book") if isinstance(container, dict) else None
    if not isinstance(book, dict):
        book = None
    for key in keys:
        for scope in (container, book):
            if not isinstance(scope, dict):
                continue
            value = scope.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return parsed
    return None


def _quantiles_for(tf_minutes: float, market_type: str | None) -> tuple[float, float]:
    """Return (volatility_quantile, multiplier_quantile) pair.

    The thresholds adapt to the timeframe in minutes and apply a small bias
    depending on the market microstructure.  Derivatives markets generally
    exhibit higher volatility, so the helper lowers the volatility quantile and
    raises the multiplier quantile to keep signals selective.  Unknown market
    types default to a mild adjustment compared to spot instruments.
    """

    minutes = max(1.0, float(tf_minutes))
    if minutes <= 2.0:
        base_vol = 0.30
        base_mult = 0.70
    elif minutes <= 5.0:
        base_vol = 0.25
        base_mult = 0.75
    elif minutes <= 30.0:
        base_vol = 0.22
        base_mult = 0.78
    elif minutes <= 60.0:
        base_vol = 0.20
        base_mult = 0.82
    else:
        base_vol = 0.18
        base_mult = 0.85

    market = (market_type or "spot").strip().lower()
    if market in {"perp", "perps", "perpetual", "futures", "swap", "derivatives"}:
        vol_adj = -0.02
        mult_adj = 0.02
    elif market in {"spot"}:
        vol_adj = 0.0
        mult_adj = 0.0
    else:
        vol_adj = -0.01
        mult_adj = 0.01

    vol_q = max(0.05, min(0.95, base_vol + vol_adj))
    mult_q = max(0.1, min(0.95, base_mult + mult_adj))
    return vol_q, mult_q


class BreakoutVol(Strategy):
    """Ruptura por volatilidad con umbrales autocalibrados.

    El multiplicador del canal y la volatilidad mínima se estiman como
    percentiles recientes, de modo que la estrategia se adapta a las
    condiciones cambiantes del mercado sin necesidad de ajustar parámetros
    manualmente.  Los precios límite se sitúan en puntos básicos respecto al
    último precio observado y se acotan por porcentaje máximo para evitar
    deslizamientos extremos.
    """

    name = "breakout_vol"

    # Percentiles dependientes del timeframe para mínimos de volatilidad y
    # dimensionar el multiplicador.  Valores más bajos en timeframes cortos
    # permiten generar más señales sin relajar el control de riesgo.
    def __init__(self, **kwargs):
        self.risk_service = kwargs.get("risk_service")
        tf = str(kwargs.get("timeframe", "3m"))
        self.timeframe = tf
        tf_minutes = timeframe_to_minutes(tf)
        self.market_type = kwargs.get("market_type")
        self._vol_quantile, self._mult_quantile = _quantiles_for(tf_minutes, self.market_type)
        self.volatility_factor = float(kwargs.get("volatility_factor", 0.02))
        self.max_offset_pct = max(0.0, float(kwargs.get("max_offset_pct", 0.015)))
        self._lookback_minutes = float(kwargs.get("lookback", 10))
        self.base_lookback = self._lookback_minutes
        self._volume_ma_minutes = float(kwargs.get("volume_ma_n", 20))
        self.base_volume_ma = self._volume_ma_minutes
        cooldown_param = kwargs.get("cooldown_bars")
        if cooldown_param is None:
            cooldown_param = 3.0
        self._cooldown_minutes = float(cooldown_param)
        self.cooldown_bars = self._cooldown_for(tf_minutes)
        self._cooldown = 0
        self._last_rpnl = 0.0
        self._last_month: int | None = None
        # Valores dinámicos calculados en ``on_bar``.
        self.mult = 1.0
        self.min_volatility = 0.0
        self._rq = RollingQuantileCache()

    def _cooldown_for(self, tf_minutes: float) -> int:
        if self._cooldown_minutes <= 0:
            return 0
        return max(1, int(math.ceil(self._cooldown_minutes / max(tf_minutes, 1e-9))))

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf_val = bar.get("timeframe", self.timeframe)
        tf_minutes = timeframe_to_minutes(tf_val)
        lookback = max(2, int(math.ceil(self._lookback_minutes / tf_minutes)))
        vol_ma_n = max(1, int(math.ceil(self._volume_ma_minutes / tf_minutes)))
        self.cooldown_bars = self._cooldown_for(tf_minutes)
        if len(df) < lookback + 1:
            return None

        if self.risk_service is not None:
            rpnl = getattr(self.risk_service.pos, "realized_pnl", 0.0)
            if rpnl < self._last_rpnl and abs(self.risk_service.pos.qty) < 1e-9:
                self._cooldown = self.cooldown_bars
            self._last_rpnl = rpnl
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        closes = df["close"]
        mean = closes.rolling(lookback).mean().iloc[-1]
        std_series = closes.rolling(lookback).std().dropna()
        std = float(std_series.iloc[-1])
        window = min(len(std_series), lookback * 5)
        symbol = bar.get("symbol", "")
        if window >= lookback and std > 0.0:
            rq_std = self._rq.get(
                symbol,
                "std",
                window=lookback * 5,
                q=self._mult_quantile,
                min_periods=lookback,
            )
            mult_quant = float(rq_std.update(std))
            self.mult = mult_quant / std
        else:
            self.mult = 1.0

        last = float(closes.iloc[-1])
        abs_last = max(abs(last), 1e-9)
        upper = mean + self.mult * std
        lower = mean - self.mult * std

        high = df.get("high")
        low = df.get("low")
        atr_val = 0.0
        if high is not None and low is not None:
            prev_close = closes.shift()
            tr = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)
            atr_series = tr.rolling(lookback).mean().dropna()
            atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0

        returns = closes.pct_change().dropna()
        vol_series = returns.rolling(lookback).std().dropna()
        vol = float(vol_series.iloc[-1]) if len(vol_series) else 0.0
        vol_bps = vol * 10000
        window = min(len(vol_series), lookback * 5)
        target_vol = abs_last * vol
        if window >= lookback * 2:
            rq_vol = self._rq.get(
                symbol,
                "vol",
                window=lookback * 5,
                q=self._vol_quantile,
                min_periods=lookback * 2,
            )
            vol_quant = float(rq_vol.update(vol))
            rq_vol_bps = self._rq.get(
                symbol,
                "vol_bps",
                window=lookback * 5,
                q=self._vol_quantile,
                min_periods=lookback * 2,
            )
            vol_bps_quant = float(rq_vol_bps.update(vol_bps))
            self.min_volatility = vol_bps_quant
            slack_vol = 0.85 * vol_quant if math.isfinite(vol_quant) else 0.0
            slack_bps = 0.85 * vol_bps_quant if math.isfinite(vol_bps_quant) else 0.0
            if (
                slack_vol > 0 and vol < slack_vol
            ) or (
                slack_bps > 0 and vol_bps < slack_bps
            ):
                return None
            # Ajuste dinámico del multiplicador cuando la volatilidad es baja
            low_vol_threshold = vol_quant * 1.5
            if vol < low_vol_threshold and self.mult > 0.0:
                factor = max(0.5, vol / low_vol_threshold)
                self.mult *= factor
            self.mult = max(0.5, self.mult)
            target_base = vol_quant if math.isfinite(vol_quant) and vol_quant > 0 else vol
            target_vol = abs_last * max(target_base, vol)
        else:
            self.min_volatility = 0.0
        bar["volatility"] = abs_last * vol
        bar["target_volatility"] = max(target_vol, 0.0)

        size_raw = max(0.0, vol_bps * self.volatility_factor)
        size = min(1.0, size_raw)

        vol_ma = df["volume"].rolling(vol_ma_n).mean().iloc[-1]
        self.vol_ma_n = vol_ma_n
        self.lookback = lookback
        tf_str = str(tf_val)
        if vol_ma > 0 and df["volume"].iloc[-1] < 0.8 * vol_ma:
            return self.finalize_signal(bar, last, None)

        side: str | None = None
        if last > upper:
            side = "buy"
        elif last < lower:
            side = "sell"
        if side is None:
            return self.finalize_signal(bar, last, None)
        sig = Signal(side, size)
        best_bid = _safe_first_float(bar, ("best_bid", "bid", "bid_price", "bid_px"))
        best_ask = _safe_first_float(bar, ("best_ask", "ask", "ask_price", "ask_px"))
        spread = None
        if best_bid is not None and best_ask is not None and best_ask > best_bid:
            spread = best_ask - best_bid
        buffer_factor = 0.5
        atr_bps = (atr_val / abs_last) * 10000.0 if atr_val > 0 else 0.0
        offset_basis_bps = max(vol_bps, atr_bps)
        if offset_basis_bps <= 0.0:
            std_bps = (std / abs_last) * 10000.0 if std > 0 else 0.0
            offset_basis_bps = std_bps if std_bps > 0 else 10.0
        offset_bps = offset_basis_bps * buffer_factor
        max_offset_pct = max(0.0, float(self.max_offset_pct))
        if max_offset_pct > 0.0:
            offset_bps = min(offset_bps, max_offset_pct * 10000.0)
        offset = abs_last * (offset_bps / 10000.0)
        if offset <= 0.0:
            offset = abs_last * 0.0001
        bar_context = bar.setdefault("context", {})
        base_anchor = upper if sig.side == "buy" else lower
        anchor_source = "channel"
        if sig.side == "buy" and best_ask is not None:
            anchored = min(base_anchor, best_ask)
            if anchored == best_ask:
                anchor_source = "book"
            base_anchor = anchored
        elif sig.side == "sell" and best_bid is not None:
            anchored = max(base_anchor, best_bid)
            if anchored == best_bid:
                anchor_source = "book"
            base_anchor = anchored

        if sig.side == "buy":
            limit_price = base_anchor - offset
            if best_ask is not None:
                limit_price = min(limit_price, math.nextafter(best_ask, float("-inf")))
            limit_price = max(0.0, limit_price)
        else:
            limit_price = base_anchor + offset
            if best_bid is not None:
                limit_price = max(limit_price, math.nextafter(best_bid, float("inf")))
            limit_price = max(0.0, limit_price)

        limit_offset = abs(base_anchor - limit_price)
        limit_offset_pct = limit_offset / abs_last if abs_last > 0 else 0.0
        limit_offset_bps = limit_offset_pct * 10000.0
        bar["limit_offset"] = limit_offset
        bar["limit_offset_bps"] = limit_offset_bps
        bar["limit_offset_pct"] = limit_offset_pct
        bar_context["limit_offset"] = limit_offset
        bar_context["limit_offset_bps"] = limit_offset_bps
        bar_context["limit_offset_pct"] = limit_offset_pct
        base_price = base_anchor
        sig.limit_price = limit_price
        sig.post_only = True
        sig.metadata.update(
            {
                "base_price": base_price,
                "limit_offset": limit_offset,
                "max_offset": (
                    min(
                        abs(abs_last * max_offset_pct)
                        if max_offset_pct > 0.0
                        else limit_offset * 3,
                        spread,
                    )
                    if spread is not None and spread > 0.0
                    else (
                        abs(abs_last * max_offset_pct)
                        if max_offset_pct > 0.0
                        else limit_offset * 3
                    )
                ),
                "step_mult": 0.5,
                "chase": True,
                "anchor_source": anchor_source,
            }
        )
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(
                size,
                last,
                volatility=abs_last * vol,
                target_volatility=bar.get("target_volatility"),
                clamp=True,
            )
            stop = self.risk_service.initial_stop(last, side, vol)
            self.trade = {
                "side": side,
                "entry_price": last,
                "qty": qty,
                "stop": stop,
                "atr": vol,
                "target_volatility": bar.get("target_volatility"),
                "atr_price": atr_val,
                "strength": size,
            }

        symbol = bar.get("symbol", "")
        if self.timeframe in {"15m", "30m"} and self.risk_service is not None and symbol:
            trade = self.risk_service.get_trade(symbol)
            if trade:
                entry = trade.get("entry_price")
                atr = trade.get("atr_price", atr_val)
                if entry is not None and atr:
                    if trade.get("side") == "buy" and last < entry - 3 * atr:
                        return self.finalize_signal(bar, last, Signal("sell", 1.0))
                    if trade.get("side") == "sell" and last > entry + 3 * atr:
                        return self.finalize_signal(bar, last, Signal("buy", 1.0))
            ts = pd.to_datetime(bar.get("timestamp") or df.index[-1])
            month = ts.month
            if self._last_month is None:
                self._last_month = month
            elif month != self._last_month:
                self._last_month = month
                if trade:
                    side_exit = "sell" if trade.get("side") == "buy" else "buy"
                    return self.finalize_signal(bar, last, Signal(side_exit, 1.0))

        return self.finalize_signal(bar, last, sig)
