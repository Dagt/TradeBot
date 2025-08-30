import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi, returns


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "threshold": "Nivel de RSI para generar señal",
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
    rsi_threshold : float, optional
        Level above which a ``buy`` signal is produced (and mirrored for
        ``sell``), by default ``60``.
    """

    name = "momentum"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 55.0)
        # Optional market activity filters
        self.min_volume = kwargs.get("min_volume")
        self.min_volatility = kwargs.get("min_volatility")
        self.vol_window = kwargs.get("vol_window", 20)

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 2:
            return None

        closes = df["close"]
        price = closes.iloc[-1]
        rsi_series = rsi(df, self.rsi_n)
        prev_rsi = rsi_series.iloc[-2]
        last_rsi = rsi_series.iloc[-1]

        # Optional inactivity filters
        if self.min_volume is not None:
            if "volume" not in df or df["volume"].iloc[-1] < self.min_volume:
                return None
        if self.min_volatility is not None:
            vol = returns(df).rolling(self.vol_window).std().iloc[-1]
            if pd.isna(vol) or vol < self.min_volatility:
                return None

        upper = self.threshold
        lower = 100 - self.threshold
        if prev_rsi <= upper and last_rsi > upper:
            return Signal("buy", 1.0)
        if prev_rsi >= lower and last_rsi < lower:
            return Signal("sell", 1.0)
        return None


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
