import math
import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, returns, atr
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager


liquidity = LiquidityFilterManager()

PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "min_volume": "Volumen mínimo requerido",
    "min_volatility": "Volatilidad mínima requerida",
    "vol_window": "Ventana para estimar la volatilidad",
    "cooldown_bars": "Barras a esperar tras una pérdida",
}


class Momentum(Strategy):
    """Simple momentum strategy using the Relative Strength Index (RSI).

    Parameters are provided via ``**kwargs`` so the class can be easily
    instantiated from configuration dictionaries.

    Parameters
    ----------
    rsi_n : int, optional
        Lookback window for the RSI calculation, by default ``14``.
    """

    name = "momentum"

    def __init__(self, **kwargs):
        self.fast_ema = kwargs.get("fast_ema", 10)
        self.slow_ema = kwargs.get("slow_ema", 30)
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.atr_n = kwargs.get("atr_n", 14)
        self.use_rsi = kwargs.get("use_rsi", True)
        # Always enable ROC confirmation
        self.use_roc = True
        self.roc_n = kwargs.get("roc_n", 10)
        # Optional market activity filters
        self.min_volume = kwargs.get("min_volume")
        self.min_volatility = kwargs.get("min_volatility")
        self.vol_window = kwargs.get("vol_window", 20)
        self.risk_service = kwargs.get("risk_service")
        # Cooldown management after losing trades
        tf = kwargs.get("timeframe", "1m")
        self.cooldown_bars = kwargs.get("cooldown_bars", 3 if tf in {"1m", "3m"} else 0)
        self._cooldown = 0
        self._last_rpnl = 0.0
        self._rq = RollingQuantileCache()

    def _tf_to_minutes(self, tf: str | None) -> int:
        """Convert timeframe strings like ``1m`` or ``15m`` to minutes."""

        if not tf:
            return 1
        tf = tf.lower()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 60 * 24
        return 1

    def auto_threshold(self, symbol: str, last_rsi: float, n: int) -> float:
        """Automatically derive RSI threshold from recent values.

        Maintains an incremental 75th percentile of the RSI using
        :class:`~tradingbot.utils.rolling_quantile.RollingQuantile`.  This
        avoids recalculating the quantile over the entire window for every new
        bar.
        """

        rq = self._rq.get(symbol, "rsi", window=n, q=0.75)
        thresh = rq.update(float(last_rsi))
        return 55.0 if pd.isna(thresh) else float(thresh)

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]

        tf_min = self._tf_to_minutes(bar.get("timeframe"))
        rsi_n = max(5, int(math.ceil(self.rsi_n / tf_min)))
        atr_n = max(5, int(math.ceil(self.atr_n / tf_min)))
        vol_window = max(2, int(math.ceil(self.vol_window / tf_min)))

        if len(df) < max(slow_n, rsi_n, atr_n) + 2:
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
        price = float(closes.iloc[-1])

        # --- Manage existing trade
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

        rsi_series = rsi(df, rsi_n)
        rsi_now = float(rsi_series.iloc[-1])

        atr_series = atr(df, atr_n)
        atr_val = float(atr_series.iloc[-1])
        bar["atr"] = atr_val

        # Optional inactivity filters
        min_vol = self.min_volume
        symbol = bar.get("symbol", "")
        if min_vol is None and "volume" in df:
            rq_vol = self._rq.get(
                symbol,
                "volume",
                window=vol_window,
                q=0.2,
                min_periods=1,
            )
            min_vol = float(rq_vol.update(float(df["volume"].iloc[-1])))
        if min_vol is not None:
            if "volume" not in df or df["volume"].iloc[-1] < min_vol:
                return None

        ret = returns(df)
        vol_series = ret.rolling(vol_window).std()
        vol = float(vol_series.iloc[-1])
        min_volatility = self.min_volatility
        if min_volatility is None and not vol_series.dropna().empty:
            rq_vola = self._rq.get(
                symbol,
                "volatility",
                window=vol_window,
                q=0.2,
                min_periods=1,
            )
            min_volatility = float(rq_vola.update(vol))
        if min_volatility is not None and (pd.isna(vol) or vol < min_volatility):
            return None

        # --- Trend and regime filter
        ema_fast = closes.ewm(span=100, adjust=False).mean()
        ema_slope_ok = float(ema_fast.iloc[-1]) > float(ema_fast.iloc[-2])
        adx_ok = True
        if "adx" in df.columns:
            adx_ok = float(df["adx"].iloc[-1]) > 20
        if not (ema_slope_ok and adx_ok):
            return None

        # --- Dynamic RSI threshold
        vol_bps = (
            atr_val / price * 10000 if price else vol * 10000
        )
        thr = min(70.0, 50.0 + 0.5 * vol_bps / 1.0)
        if rsi_now < thr and rsi_now > (100 - thr):
            return None
        side = "buy" if rsi_now >= thr else "sell"

        # --- Microstructure guard for lower timeframes
        spread_bps = 0.0
        if {"bid", "ask"}.issubset(df.columns):
            spread = float(df["ask"].iloc[-1] - df["bid"].iloc[-1])
            spread_bps = spread / price * 10000 if price else 0.0
        else:
            high = float(df["high"].iloc[-1]) if "high" in df.columns else price
            low = float(df["low"].iloc[-1]) if "low" in df.columns else price
            spread_bps = (high - low) / price * 10000 if price else 0.0
        bar["spread_bps"] = spread_bps
        bar["vol_bps"] = vol_bps
        tf = bar.get("timeframe", "")
        if tf in ("1m", "3m", "5m", "15m"):
            if spread_bps > 5 or vol_bps < 8:
                return None
        if tf == "1m":
            setattr(self, "prefer_post_only", True)

        if symbol:
            if not hasattr(self, "_last_atr"):
                self._last_atr: dict[str, float] = {}
            self._last_atr[symbol] = atr_val

        strength = 1.0
        sig = Signal(side, strength)

        if self.risk_service is not None:
            qty = self.risk_service.calc_position_size(strength, price)
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
                "qty": strength,
                "stop": price - 1.5 * atr_val if side == "buy" else price + 1.5 * atr_val,
                "atr": atr_val,
                "bars_held": 0,
            }

        return self.finalize_signal(bar, price, sig)


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate momentum signals for backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Price data with a ``price`` column.
    params : dict
        Parameters including ``window``, ``position_size``, ``fee`` y ``slippage``.

    Returns
    -------
    pd.DataFrame
        Data con señal, posición y estimaciones de costos de transacción.
    """

    df = data.copy()
    window = params.get("window", 14)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)

    ma = df["price"].rolling(window).mean()
    df["signal"] = 0
    df.loc[df["price"] > ma, "signal"] = 1
    df.loc[df["price"] < ma, "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "fee", "slippage"]]
