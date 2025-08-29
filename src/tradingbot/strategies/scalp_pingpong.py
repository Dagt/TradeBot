from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, Signal, record_signal_metrics


@dataclass
class ScalpPingPongConfig:
    """Configuration for :class:`ScalpPingPong`.

    Parameters
    ----------
    lookback : int, optional
        Window length for the z-score calculation, by default ``15``.
    z_threshold : float, optional
        Absolute z-score value required to open a trade, by default ``0.2``.
    exit_z : float, optional
        Threshold below which the position is exited for mean reversion,
        by default ``0.05``.
    tp_bps : float, optional
        Take profit in basis points, by default ``10``.
    sl_bps : float, optional
        Stop loss in basis points, by default ``15``.
    max_hold_bars : int, optional
        Maximum number of bars to hold a trade, by default ``8``.
    trailing_stop_bps : float, optional
        Distance from the best price in basis points to trigger a trailing stop,
        default ``10``.
    volatility_factor : float, optional
        Multiplier applied to recent volatility (in bps) to size positions,
        default ``0.02``.
    """

    lookback: int = 15
    z_threshold: float = 0.2
    exit_z: float = 0.05
    tp_bps: float = 10.0
    sl_bps: float = 15.0
    max_hold_bars: int = 8
    trailing_stop_bps: float | None = 10.0
    volatility_factor: float = 0.02


class ScalpPingPong(Strategy):
    """Mean-reversion scalping strategy using z-score of returns."""

    name = "scalp_pingpong"

    def __init__(self, cfg: ScalpPingPongConfig | None = None, **kwargs):
        self.cfg = cfg or ScalpPingPongConfig(**kwargs)
        self.pos_side: int = 0  # 0 flat, +1 long, -1 short
        self.entry_price: float | None = None
        self.hold_bars: int = 0
        self.favorable_price: float | None = None

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

        vol = returns.rolling(self.cfg.lookback).std().iloc[-1] if len(returns) >= self.cfg.lookback else 0.0
        vol_bps = vol * 10000
        vol_size = max(0.0, min(1.0, vol_bps * self.cfg.volatility_factor))

        if self.pos_side == 0:
            if z <= -self.cfg.z_threshold:
                self.pos_side = 1
                self.entry_price = price
                self.favorable_price = price
                self.hold_bars = 0
                strength = min(1.0, abs(z) / self.cfg.z_threshold)
                size = min(1.0, strength * vol_size)
                if size > 0:
                    return Signal("buy", size)
            if z >= self.cfg.z_threshold:
                self.pos_side = -1
                self.entry_price = price
                self.favorable_price = price
                self.hold_bars = 0
                strength = min(1.0, abs(z) / self.cfg.z_threshold)
                size = min(1.0, strength * vol_size)
                if size > 0:
                    return Signal("sell", size)
            return None

        self.hold_bars += 1
        assert self.entry_price is not None and self.favorable_price is not None
        if self.pos_side > 0:
            self.favorable_price = max(self.favorable_price, price)
        else:
            self.favorable_price = min(self.favorable_price, price)

        pnl_bps = (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
        exit_z = abs(z) < self.cfg.exit_z
        exit_tp = pnl_bps >= self.cfg.tp_bps
        exit_sl = pnl_bps <= -self.cfg.sl_bps
        exit_time = self.hold_bars >= self.cfg.max_hold_bars
        exit_trail = False
        if self.cfg.trailing_stop_bps is not None:
            best_pnl = (price - self.favorable_price) / self.favorable_price * 10000 * self.pos_side
            exit_trail = best_pnl <= -self.cfg.trailing_stop_bps
        if exit_z or exit_tp or exit_sl or exit_time or exit_trail:
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.favorable_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None
