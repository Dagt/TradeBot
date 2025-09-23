import math

import math
import pandas as pd
from .base import Strategy, Signal, record_signal_metrics, timeframe_to_minutes
from ..data.features import rsi, calc_ofi
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "min_volatility": "Volatilidad mínima requerida",
    "vol_lookback": "Ventana para calcular la volatilidad (minutos)",
}


liquidity = LiquidityFilterManager()


class TrendFollowing(Strategy):
    """RSI based trend following strategy.

    Signals are generated when the RSI crosses extreme levels. Risk management,
    including position sizing and exits, is delegated to the universal
    ``RiskManager`` outside this strategy.
    """

    name = "trend_following"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        # ``vol_lookback`` se especifica en minutos y se escalará al timeframe
        # real en ``on_bar``.
        self.vol_lookback = kwargs.get("vol_lookback", self.rsi_n)
        self._min_volatility_override = kwargs.get("min_volatility")
        if self._min_volatility_override is not None:
            self.min_volatility = float(self._min_volatility_override)
        else:
            self.min_volatility = 0.0
        self.risk_service = kwargs.get("risk_service")
        self.timeframe = str(kwargs.get("timeframe", "1m"))
        self._rq = RollingQuantileCache()

    # ------------------------------------------------------------------
    @staticmethod
    def _tf_minutes(tf: str | None, default: str | None = None) -> int:
        """Convert timeframe strings like ``'1m'`` or ``'15m'`` to minutes."""

        if tf is None:
            tf = default
        minutes = timeframe_to_minutes(tf)
        return max(1, int(math.ceil(minutes)))

    def auto_threshold(self, symbol: str, last_rsi: float, vol_bps: float) -> float:
        """Derive RSI threshold based on recent volatility.

        The threshold starts at the rolling median RSI and shifts by a
        multiplier of the current volatility expressed in basis points.  A
        higher volatility therefore raises the entry level making signals more
        selective when markets are noisy.
        """

        rq = self._rq.get(symbol, "rsi_median", window=self.rsi_n, q=0.5)
        base = rq.update(last_rsi)
        base = 50.0 if pd.isna(base) else float(base)
        thresh = base + vol_bps * 0.5
        return max(55.0, min(90.0, thresh))

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        tf = bar.get("timeframe", self.timeframe)
        tf_minutes = self._tf_minutes(tf, self.timeframe)
        lookback_bars = max(1, math.ceil(self.vol_lookback / tf_minutes))
        if len(df) < max(self.rsi_n, lookback_bars) + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        prices = df[price_col]
        price = float(prices.iloc[-1])
        returns = prices.pct_change().dropna()
        vol_series = returns.rolling(lookback_bars).std().dropna()
        vol_bps = float(vol_series.iloc[-1]) * 10000 if len(vol_series) else 0.0
        window = min(len(vol_series), lookback_bars * 5)
        symbol = bar.get("symbol", "")
        if window >= lookback_bars:
            rq_vol = self._rq.get(
                symbol,
                "vol_bps",
                window=lookback_bars * 5,
                q=0.2,
                min_periods=lookback_bars,
            )
            val = rq_vol.update(vol_bps)
            if self._min_volatility_override is None:
                self.min_volatility = 0.0 if pd.isna(val) else float(val)
        elif self._min_volatility_override is None:
            self.min_volatility = 0.0
        if pd.isna(vol_bps) or vol_bps < self.min_volatility:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = float(rsi_series.iloc[-1])
        threshold = self.auto_threshold(symbol, last_rsi, vol_bps)
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        side: str | None = None
        base_strength = 0.0
        if last_rsi > threshold:
            side = "buy"
            base_strength = (last_rsi - threshold) / max(1.0, 100.0 - threshold)
        elif last_rsi < 100 - threshold:
            side = "sell"
            lower_threshold = 100.0 - threshold
            base_strength = (lower_threshold - last_rsi) / max(1.0, lower_threshold)
        if side is None:
            return None
        base_strength = max(0.0, min(1.0, base_strength))
        direction = 1.0 if side == "buy" else -1.0
        if ofi_val == 0:
            ofi_factor = 1.0
        else:
            ofi_factor = max(0.0, math.copysign(1.0, ofi_val) * direction)
            if ofi_factor == 0.0:
                return None
        strength = max(0.0, min(1.0, base_strength * ofi_factor))
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
