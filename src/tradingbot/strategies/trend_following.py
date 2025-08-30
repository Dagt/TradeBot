import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, calc_ofi


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "threshold": "Nivel de RSI para señales de tendencia",
    "tp_bps": "Take profit en puntos básicos",
    "sl_bps": "Stop loss en puntos básicos",
    "max_hold_bars": "Barras máximas en posición",
    "min_volatility": "Volatilidad mínima requerida",
    "vol_lookback": "Ventana para calcular la volatilidad",
}


class TrendFollowing(Strategy):
    """RSI based trend following strategy with adaptive strength.

    Signals are generated when the RSI crosses extreme levels.  The returned
    ``strength`` scales up if an existing position is profitable and the new
    signal aligns with it.  Adverse moves reduce the strength and may turn it
    negative to indicate that the position should be reduced or flipped.
    Additionally, the strategy supports basic take-profit/stop-loss/time based
    exits and requires a minimum volatility to operate.
    """

    name = "trend_following"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 60.0)
        self.tp_bps = kwargs.get("tp_bps", 100.0)
        self.sl_bps = kwargs.get("sl_bps", 50.0)
        self.max_hold_bars = kwargs.get("max_hold_bars", 30)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.vol_lookback = kwargs.get("vol_lookback", self.rsi_n)
        self._pos_side: str | None = None
        self._entry_price: float | None = None
        self.hold_bars = 0

    def _calc_strength(self, side: str, price: float) -> float:
        if side == "flat":
            self._pos_side = None
            self._entry_price = None
            self.hold_bars = 0
            return 0.0
        strength = 1.0
        if self._pos_side and self._entry_price:
            pnl_bps = (price - self._entry_price) / self._entry_price * 10000
            if self._pos_side == "sell":
                pnl_bps = -pnl_bps
            scale = self.tp_bps if self.tp_bps else 1.0
            if side == self._pos_side:
                strength += pnl_bps / scale
            else:
                strength = -pnl_bps / scale
        if strength > 0:
            self._pos_side = side
            self._entry_price = price
            self.hold_bars = 0
        else:
            self._pos_side = None
            self._entry_price = None
            self.hold_bars = 0
        return strength

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.rsi_n, self.vol_lookback) + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        prices = df[price_col]
        price = float(prices.iloc[-1])

        # Manage existing position first
        if self._pos_side:
            self.hold_bars += 1
            assert self._entry_price is not None
            pnl_bps = (price - self._entry_price) / self._entry_price * 10000
            if self._pos_side == "sell":
                pnl_bps = -pnl_bps
            exit_tp = pnl_bps >= self.tp_bps
            exit_sl = pnl_bps <= -self.sl_bps
            exit_time = self.hold_bars >= self.max_hold_bars
            if exit_tp or exit_sl or exit_time:
                side = "sell" if self._pos_side == "buy" else "buy"
                self._calc_strength("flat", price)
                return Signal(side, 1.0)
            return None

        # No position: compute indicators for entry
        returns = prices.pct_change().dropna()
        vol = returns.rolling(self.vol_lookback).std().iloc[-1] * 10000
        if pd.isna(vol) or vol < self.min_volatility:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if last_rsi > self.threshold and ofi_val >= 0:
            strength = self._calc_strength("buy", price)
            return Signal("buy", strength)
        if last_rsi < 100 - self.threshold and ofi_val <= 0:
            strength = self._calc_strength("sell", price)
            return Signal("sell", strength)
        self._calc_strength("flat", price)
        return None
