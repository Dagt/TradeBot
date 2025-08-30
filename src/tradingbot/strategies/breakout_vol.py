import pandas as pd

from .base import Strategy, Signal, record_signal_metrics


PARAM_INFO = {
    "lookback": "Ventana para medias y desviación estándar",
    "mult": "Multiplicador aplicado a la desviación",
    "tp_bps": "Take profit en puntos básicos",
    "sl_bps": "Stop loss en puntos básicos",
    "max_hold_bars": "Barras máximas en posición",
    "trailing_stop_bps": "Distancia del trailing stop en bps",
    "volatility_factor": "Factor para dimensionar según volatilidad",
    "min_edge_bps": "Edge mínimo en puntos básicos para operar",
    "min_volatility": "Volatilidad mínima reciente en bps",
}


class BreakoutVol(Strategy):
    """Volatility breakout strategy using rolling standard deviation.

    Parameters are supplied via ``**kwargs`` and stored as attributes.  The
    strategy computes a rolling mean and standard deviation of the close price
    and generates a ``buy`` signal when price breaks above ``mean + mult *
    std``.  A ``sell`` signal is produced for a break below ``mean - mult *
    std``.

    Parameters
    ----------
    lookback : int, optional
        Window size for the rolling statistics, default ``10``.
    mult : float, optional
        Multiplier applied to the standard deviation, default ``2``.
    tp_bps : float, optional
        Take profit in basis points, default ``10``.
    sl_bps : float, optional
        Stop loss in basis points, default ``15``.
    trailing_stop_bps : float, optional
        Distance from the best price in basis points to trigger a trailing stop,
        default ``10``.
    volatility_factor : float, optional
        Multiplier applied to recent volatility (in bps) to size positions,
        default ``0.02``.
    min_edge_bps : float, optional
        Edge mínimo en puntos básicos requerido para operar, default ``0``.
    """

    name = "breakout_vol"

    def __init__(self, **kwargs):
        self.lookback = kwargs.get("lookback", 10)
        self.mult = kwargs.get("mult", 1.5)
        self.tp_bps = kwargs.get("tp_bps", 10.0)
        self.sl_bps = kwargs.get("sl_bps", 15.0)
        self.max_hold_bars = kwargs.get("max_hold_bars", 10)
        self.trailing_stop_bps = kwargs.get("trailing_stop_bps", 10.0)
        self.volatility_factor = kwargs.get("volatility_factor", 0.02)
        self.min_edge_bps = kwargs.get("min_edge_bps", 0.0)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.favorable_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.lookback + 1:
            return None
        closes = df["close"]
        mean = closes.rolling(self.lookback).mean().iloc[-1]
        std = closes.rolling(self.lookback).std().iloc[-1]
        last = closes.iloc[-1]
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

        if self.pos_side == 0:
            if last > upper:
                expected_edge_bps = (last - upper) / abs(last) * 10000
                if expected_edge_bps <= self.min_edge_bps:
                    return None
                self.pos_side = 1
                self.entry_price = last
                self.favorable_price = last
                self.hold_bars = 0
                return Signal("buy", size, expected_edge_bps=expected_edge_bps)
            if last < lower:
                expected_edge_bps = (lower - last) / abs(last) * 10000
                if expected_edge_bps <= self.min_edge_bps:
                    return None
                self.pos_side = -1
                self.entry_price = last
                self.favorable_price = last
                self.hold_bars = 0
                return Signal("sell", size, expected_edge_bps=expected_edge_bps)
            return None

        self.hold_bars += 1
        assert self.entry_price is not None and self.favorable_price is not None
        if self.pos_side > 0:
            self.favorable_price = max(self.favorable_price, last)
        else:
            self.favorable_price = min(self.favorable_price, last)

        pnl_bps = (last - self.entry_price) / self.entry_price * 10000 * self.pos_side
        trail_hit = False
        if self.trailing_stop_bps is not None:
            best_pnl = (last - self.favorable_price) / self.favorable_price * 10000 * self.pos_side
            trail_hit = best_pnl <= -self.trailing_stop_bps
        if (
            pnl_bps >= self.tp_bps
            or pnl_bps <= -self.sl_bps
            or self.hold_bars >= self.max_hold_bars
            or trail_hit
        ):
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.favorable_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None
