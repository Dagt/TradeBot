import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
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
        self.min_volatility = 0.0
        self.risk_service = kwargs.get("risk_service")
        self._rq = RollingQuantileCache()

    # ------------------------------------------------------------------
    @staticmethod
    def _tf_minutes(tf: str | None) -> int:
        """Convert timeframe strings like ``'1m'`` or ``'15m'`` to minutes."""

        if not tf:
            return 1
        tf = tf.lower()
        if tf.endswith("m"):
            return int(tf[:-1] or 1)
        if tf.endswith("h"):
            return int(tf[:-1] or 1) * 60
        return 1

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
        tf = bar.get("timeframe")
        tf_minutes = self._tf_minutes(tf)
        lookback_bars = max(1, int(self.vol_lookback / tf_minutes))
        if len(df) < max(self.rsi_n, lookback_bars) + 1:
            return None
        price_col = "close" if "close" in df.columns else "price"
        prices = df[price_col]
        price = float(prices.iloc[-1])

        # --- Basic microstructure guard
        spread_bps = 0.0
        if {"bid", "ask"}.issubset(df.columns):
            spread = float(df["ask"].iloc[-1] - df["bid"].iloc[-1])
            spread_bps = spread / price * 10000 if price else 0.0
        else:
            high = float(df["high"].iloc[-1]) if "high" in df.columns else price
            low = float(df["low"].iloc[-1]) if "low" in df.columns else price
            spread_bps = (high - low) / price * 10000 if price else 0.0
        bar["spread_bps"] = spread_bps

        # --- Manage open trade
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
            self.min_volatility = 0.0 if pd.isna(val) else float(val)
        else:
            self.min_volatility = 0.0
        if pd.isna(vol_bps) or vol_bps < self.min_volatility:
            return None
        bar["vol_bps"] = vol_bps
        if spread_bps > 5 or vol_bps < 8:
            return None
        rsi_series = rsi(df, self.rsi_n)
        rsi_now = float(rsi_series.iloc[-1])

        ma30 = df["close"].rolling(30).mean().iloc[-1]
        dist_pct = (
            (df["close"].iloc[-1] - ma30) / ma30 * 100 if ma30 else 0
        )
        if rsi_now >= 70 or dist_pct >= 2.0:
            return None

        ema_proxy_htf = df["close"].ewm(span=400, adjust=False).mean()
        if ema_proxy_htf.iloc[-1] <= ema_proxy_htf.iloc[-2]:
            return None

        threshold = self.auto_threshold(symbol, rsi_now, vol_bps)
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if rsi_now > threshold and ofi_val >= 0:
            side = "buy"
        elif rsi_now < 100 - threshold and ofi_val <= 0:
            side = "sell"
        else:
            return None

        jump = abs(df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100
        strength = max(0.2, 1.0 - min(jump / 2.0, 0.8))
        sig = Signal(side, strength)
        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(strength, price)
            atr_val = bar.get("atr") or bar.get("volatility")
            stop = self.risk_service.initial_stop(price, side, atr_val, atr_mult=1.5)
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "bars_held": 0,
            }
        else:
            atr_val = bar.get("atr") or bar.get("volatility")
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": strength,
                "stop": price - 1.5 * atr_val if side == "buy" else price + 1.5 * atr_val,
                "atr": atr_val,
                "bars_held": 0,
            }
        return self.finalize_signal(bar, price, sig)
