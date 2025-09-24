import math
import pandas as pd
from .base import Strategy, Signal, record_signal_metrics, timeframe_to_minutes
from ..data.features import rsi, returns, atr
from ..utils.rolling_quantile import RollingQuantileCache
from ..filters.liquidity import LiquidityFilterManager


liquidity = LiquidityFilterManager()

# Minimum number of bars to compute momentum indicators reliably on higher
# timeframes.  This prevents ``fast``/``slow`` EMA and RSI windows collapsing
# to one or two observations when the configured minutes are shorter than the
# candle size (e.g. running 10-minute EMAs on 15-minute bars).
MIN_BARS = 5

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
        tf = str(kwargs.get("timeframe", "1m"))
        self.timeframe = tf
        tf_minutes = timeframe_to_minutes(tf)
        cooldown_param = kwargs.get("cooldown_bars")
        if cooldown_param is None:
            cooldown_param = 3.0
        self._cooldown_minutes = float(cooldown_param)
        self.cooldown_bars = self._cooldown_for(tf_minutes)
        self._cooldown = 0
        self._last_rpnl = 0.0
        self._rq = RollingQuantileCache()

    def _cooldown_for(self, tf_minutes: float) -> int:
        if self._cooldown_minutes <= 0:
            return 0
        return max(1, int(math.ceil(self._cooldown_minutes / max(tf_minutes, 1e-9))))

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

        tf_min = timeframe_to_minutes(bar.get("timeframe", self.timeframe))
        self.cooldown_bars = self._cooldown_for(tf_min)

        bar_minutes = max(tf_min, 1e-9)
        fast_minutes = max(float(self.fast_ema), 0.0)
        slow_minutes = max(float(self.slow_ema), fast_minutes)
        fast_n = max(MIN_BARS, int(math.ceil(fast_minutes / bar_minutes)))
        slow_n = max(MIN_BARS, int(math.ceil(slow_minutes / bar_minutes)))
        if slow_n <= fast_n:
            ratio = slow_minutes / max(fast_minutes, 1e-9)
            scaled_slow = int(math.ceil(fast_n * ratio))
            slow_n = max(fast_n + 1, scaled_slow)

        rsi_n = max(MIN_BARS, int(math.ceil(float(self.rsi_n) / bar_minutes)))
        atr_n = max(MIN_BARS, int(math.ceil(float(self.atr_n) / bar_minutes)))
        vol_window = max(MIN_BARS, int(math.ceil(float(self.vol_window) / bar_minutes)))

        if len(df) < max(slow_n, rsi_n, atr_n, vol_window) + 2:
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

        fast_ema = closes.ewm(span=fast_n, adjust=False).mean()
        slow_ema = closes.ewm(span=slow_n, adjust=False).mean()
        prev_fast, prev_slow = fast_ema.iloc[-2], slow_ema.iloc[-2]
        last_fast, last_slow = float(fast_ema.iloc[-1]), float(slow_ema.iloc[-1])
        slope = float(slow_ema.iloc[-1] - slow_ema.iloc[-2])

        rsi_series = rsi(df, rsi_n)
        last_rsi = float(rsi_series.iloc[-1])

        roc_val = float(closes.pct_change(self.roc_n).iloc[-1])

        if "high" in df.columns and "low" in df.columns:
            ohlc = df
        else:
            ohlc = df.copy()
            if "high" not in ohlc:
                ohlc["high"] = ohlc["close"]
            if "low" not in ohlc:
                ohlc["low"] = ohlc["close"]

        atr_series = atr(ohlc, atr_n)
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
        if min_volatility is not None:
            if pd.isna(vol) or vol < min_volatility:
                return None

        # Momentum rules with RSI and ROC confirmation in all timeframes
        side: str | None = None
        rsi_thresh = self.auto_threshold(symbol, last_rsi, n=rsi_n) if self.use_rsi else 50.0
        if prev_fast <= prev_slow and last_fast > last_slow:
            cond = slope > 0
            if self.use_rsi:
                cond &= last_rsi > rsi_thresh
            cond &= roc_val > 0
            if cond:
                side = "buy"
        elif prev_fast >= prev_slow and last_fast < last_slow:
            cond = slope < 0
            if self.use_rsi:
                cond &= last_rsi < rsi_thresh
            cond &= roc_val < 0
            if cond:
                side = "sell"
        if side is None:
            return None

        if symbol:
            if not hasattr(self, "_last_atr"):
                self._last_atr: dict[str, float] = {}
            self._last_atr[symbol] = atr_val

        strength = 1.0
        sig = Signal(side, strength)

        if self.risk_service is not None:
            stop_mult = 1.5 if tf_min <= 5 else 2.0
            qty = self.risk_service.calc_position_size(strength, price)
            stop = self.risk_service.initial_stop(
                price, side, atr_val, atr_mult=stop_mult
            )
            if tf_min <= 3:
                max_hold = 20
            elif tf_min <= 5:
                max_hold = 10
            else:
                max_hold = 20
            self.trade = {
                "side": side,
                "entry_price": price,
                "qty": qty,
                "stop": stop,
                "atr": atr_val,
                "bars_held": 0,
                "max_hold": max_hold,
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
