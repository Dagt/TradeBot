from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .base import Strategy, Signal, load_params, record_signal_metrics
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
        Fraction of the recent volatility (in bps or ATR) used to size
        positions, default ``0.02``.
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
        self.cfg = cfg or ScalpPingPongConfig(**params)
        self.risk_service = kwargs.get("risk_service")
        self.prefer_post_only = True

    def _calc_zscore(self, closes: pd.Series, lookback: int) -> float:
        returns = closes.pct_change().dropna()
        if len(returns) < lookback:
            return 0.0
        mean = returns.rolling(lookback).mean().iloc[-1]
        std = returns.rolling(lookback).std().iloc[-1]
        if std == 0 or pd.isna(std):
            return 0.0
        z = (returns.iloc[-1] - mean) / std
        return float(z)

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf = int(bar.get("timeframe", 1)) or 1
        lookback = max(1, int(self.cfg.lookback / tf))
        trend_ma = max(1, int(self.cfg.trend_ma / tf))
        trend_rsi_n = max(1, int(self.cfg.trend_rsi_n / tf))

        if len(df) < lookback + 1:
            return None
        closes = df["close"]
        returns = closes.pct_change().dropna()
        z = self._calc_zscore(closes, lookback)
        price = float(closes.iloc[-1])

        spread_bps = 0.0
        if {"bid", "ask"}.issubset(df.columns):
            spread = float(df["ask"].iloc[-1] - df["bid"].iloc[-1])
            spread_bps = spread / price * 10000 if price else 0.0
        else:
            high = float(df["high"].iloc[-1]) if "high" in df.columns else price
            low = float(df["low"].iloc[-1]) if "low" in df.columns else price
            spread_bps = (high - low) / price * 10000 if price else 0.0
        bar["spread_bps"] = spread_bps

        trade = getattr(self, "trade", None)
        if trade:
            trade["bars_held"] = trade.get("bars_held", 0) + 1
            take1 = trade.get("take1")
            if take1 is None:
                take1 = (
                    trade["entry_price"] + 0.5 * trade["atr"]
                    if trade["side"] == "buy"
                    else trade["entry_price"] - 0.5 * trade["atr"]
                )
                trade["take1"] = take1
            if not trade.get("tp_hit") and (
                (trade["side"] == "buy" and price >= take1)
                or (trade["side"] == "sell" and price <= take1)
            ):
                trade["tp_hit"] = True
                trade["qty"] = float(trade.get("qty", 0)) * 0.5
                sig = Signal(trade["side"], 0.5, reduce_only=True)
                return self.finalize_signal(bar, price, sig)
            if trade["bars_held"] >= 6 and not trade.get("tp_hit"):
                side = "sell" if trade["side"] == "buy" else "buy"
                sig = Signal(side, 1.0, reduce_only=True)
                self.trade = None
                return self.finalize_signal(bar, price, sig)

        if bar.get("atr") is not None and price != 0:
            vol_bps = float(bar["atr"]) / abs(price) * 10000
        else:
            vol = (
                returns.rolling(lookback).std().iloc[-1]
                if len(returns) >= lookback
                else 0.0
            )
            vol_bps = vol * 10000
        if vol_bps < self.cfg.min_volatility or spread_bps > 3 or vol_bps < 10:
            return None
        bar["vol_bps"] = vol_bps
        vol_size = max(0.0, min(1.0, vol_bps * self.cfg.volatility_factor / 10000))

        ma = closes.rolling(50).mean().iloc[-1]
        trend_bias = 0
        if not pd.isna(ma) and ma != 0:
            diff_pct = (price - ma) / ma * 100
            if diff_pct > 0.5:
                trend_bias = 1
            elif diff_pct < -0.5:
                trend_bias = -1

        z_buy = self.cfg.z_threshold + (0.1 if trend_bias == -1 else 0.0)
        z_sell = self.cfg.z_threshold + (0.1 if trend_bias == 1 else 0.0)

        if z <= -z_buy:
            side = "buy"
            strength = min(1.0, abs(z) / z_buy)
        elif z >= z_sell:
            side = "sell"
            strength = min(1.0, abs(z) / z_sell)
        else:
            return None

        size = min(
            1.0,
            strength * vol_size * max(0.2, 1.0 - spread_bps / 10.0),
        )
        if size <= 0.0:
            return self.finalize_signal(bar, price, None)
        sig = Signal(side, size)

        atr_val = bar.get("atr")
        if atr_val is None:
            atr_val = abs(price) * vol_bps / 10000

        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(size, price)
            stop = self.risk_service.initial_stop(
                price, side, atr_val, atr_mult=1.5
            )
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "bars_held": 0,
            }
        else:
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": size,
                "stop": price - 1.5 * atr_val if side == "buy" else price + 1.5 * atr_val,
                "atr": atr_val,
                "bars_held": 0,
            }

        return self.finalize_signal(bar, price, sig)
