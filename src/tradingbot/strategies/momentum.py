import math
import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, returns
from ..utils.rolling_quantile import RollingQuantileCache


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "min_volume": "Volumen mínimo requerido",
    "min_volatility": "Volatilidad mínima requerida",
    "vol_window": "Ventana para estimar la volatilidad",
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
        self.rsi_n = kwargs.get("rsi_n", 14)
        # Optional market activity filters
        self.min_volume = kwargs.get("min_volume")
        self.min_volatility = kwargs.get("min_volatility")
        self.vol_window = kwargs.get("vol_window", 20)
        self.risk_service = kwargs.get("risk_service")
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

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]

        tf_min = self._tf_to_minutes(bar.get("timeframe"))
        rsi_n = max(2, int(math.ceil(self.rsi_n / tf_min)))
        vol_window = max(2, int(math.ceil(self.vol_window / tf_min)))

        if len(df) < rsi_n + 2:
            return None

        closes = df["close"]
        price = float(closes.iloc[-1])
        rsi_series = rsi(df, rsi_n)
        prev_rsi = rsi_series.iloc[-2]
        last_rsi = float(rsi_series.iloc[-1])

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

        threshold = self.auto_threshold(symbol, last_rsi, rsi_n)
        upper = threshold
        lower = 100 - threshold
        side: str | None = None
        if prev_rsi <= upper and last_rsi > upper:
            side = "buy"
        elif prev_rsi >= lower and last_rsi < lower:
            side = "sell"
        if side is None:
            return None
        strength = 1.0
        sig = Signal(side, strength)
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
