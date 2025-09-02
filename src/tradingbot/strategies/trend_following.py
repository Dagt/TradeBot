import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, calc_ofi


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "min_volatility": "Volatilidad mínima requerida",
    "vol_lookback": "Ventana para calcular la volatilidad",
}


class TrendFollowing(Strategy):
    """RSI based trend following strategy.

    Signals are generated when the RSI crosses extreme levels. Risk management,
    including position sizing and exits, is delegated to the universal
    ``RiskManager`` outside this strategy.
    """

    name = "trend_following"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.min_volatility = kwargs.get("min_volatility", 0.0)
        self.vol_lookback = kwargs.get("vol_lookback", self.rsi_n)
        self.risk_service = kwargs.get("risk_service")

    def auto_threshold(self, rsi_series: pd.Series, vol: float) -> float:
        """Derive RSI threshold based on recent volatility.

        Uses the median RSI as a base and adds a fraction of current
        volatility (in bps) to adapt entry levels dynamically.
        """

        base = rsi_series.rolling(self.rsi_n).median().iloc[-1]
        base = 50.0 if pd.isna(base) else float(base)
        thresh = base + vol * 0.1
        return max(55.0, min(90.0, thresh))

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
        if pd.isna(vol) or vol < self.min_volatility:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        threshold = self.auto_threshold(rsi_series, vol)
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if last_rsi > threshold and ofi_val >= 0:
            side = "buy"
        elif last_rsi < 100 - threshold and ofi_val <= 0:
            side = "sell"
        else:
            return None
        strength = 1.0
        sig = Signal(side, strength)
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(strength, price)
            atr_val = bar.get("atr") or bar.get("volatility")
            stop = self.risk_service.initial_stop(price, side, atr_val)
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
            }
        return self.finalize_signal(bar, price, sig)
