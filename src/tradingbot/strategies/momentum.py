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

    def __init__(self, risk_service=None, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 55.0)
        # Optional market activity filters
        self.min_volume = kwargs.get("min_volume")
        self.min_volatility = kwargs.get("min_volatility")
        self.vol_window = kwargs.get("vol_window", 20)
        self.risk_service = risk_service
        self.trade: dict | None = None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 2:
            return None

        closes = df["close"]
        price = float(closes.iloc[-1])
        if self.trade and self.risk_service:
            self.risk_service.update_trailing(self.trade, price)
            trade = {**self.trade, "current_price": price}
            decision = self.risk_service.manage_position(trade)
            self.trade.update(trade)
            if decision == "close":
                side = "sell" if self.trade["side"] == "buy" else "buy"
                self.trade = None
                return Signal(side, 1.0)
            if decision in {"scale_in", "scale_out"}:
                return Signal(self.trade["side"], self.trade.get("strength", 1.0))
            return None
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
        side: str | None = None
        if prev_rsi <= upper and last_rsi > upper:
            side = "buy"
        elif prev_rsi >= lower and last_rsi < lower:
            side = "sell"
        if side is None:
            return None
        strength = 1.0
        if self.risk_service:
            qty = self.risk_service.calc_position_size(strength, price)
            trade = {"side": side, "entry_price": price, "qty": qty}
            atr = bar.get("atr") or bar.get("volatility") or 0.0
            trade["stop"] = self.risk_service.initial_stop(price, side, atr)
            trade["atr"] = atr
            self.risk_service.update_trailing(trade, price)
            self.trade = trade
        return Signal(side, strength)


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
