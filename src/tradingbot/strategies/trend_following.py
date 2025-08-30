import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, calc_ofi


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "threshold": "Nivel de RSI para señales de tendencia",
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

    def __init__(self, risk_service=None, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 60.0)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.vol_lookback = kwargs.get("vol_lookback", self.rsi_n)
        self.risk_service = risk_service
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < max(self.rsi_n, self.vol_lookback) + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        prices = df[price_col]
        price = float(prices.iloc[-1])
        returns = prices.pct_change().dropna()
        vol = returns.rolling(self.vol_lookback).std().iloc[-1] * 10000
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, price)
            decision = self.risk_service.manage_position(
                {**self.trade, "current_price": price}
            )
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            return None
        if pd.isna(vol) or vol < self.min_volatility:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if last_rsi > self.threshold and ofi_val >= 0:
            side = "buy"
        elif last_rsi < 100 - self.threshold and ofi_val <= 0:
            side = "sell"
        else:
            return None
        strength = 1.0
        if self.risk_service:
            qty = self.risk_service.calc_position_size(strength, price)
            trade = {"side": side, "entry_price": price, "qty": qty}
            atr = bar.get("atr") or bar.get("volatility")
            trade["stop"] = self.risk_service.initial_stop(
                price, side, atr
            )
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        return Signal(side, strength)
