import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi

class MeanReversion(Strategy):
    """RSI based mean reversion strategy with adaptive strength.

    Besides generating ``buy``/``sell``/``flat`` signals based on RSI levels,
    the strategy adjusts the ``strength`` of those signals according to the
    performance of the current position: if price has moved in favour of the
    open trade the strength is increased (pyramiding), while adverse moves
    reduce it and can even flip it negative to suggest position reduction or
    reversal.

    Parameters are accepted through ``**kwargs`` for easy configuration.

    Parameters
    ----------
    rsi_n : int, optional
        Lookback window for the RSI calculation, by default ``14``.
    upper : float, optional
        Upper RSI level above which a ``sell`` signal is triggered, by default
        ``70``.
    lower : float, optional
        Lower RSI level below which a ``buy`` signal is triggered, by default
        ``30``.
    """

    name = "mean_reversion"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.upper = kwargs.get("upper", 60.0)
        self.lower = kwargs.get("lower", 40.0)
        # Track current position to adapt strength
        self._pos_side: str | None = None
        self._entry_price: float | None = None

    def _calc_strength(self, side: str, price: float) -> float:
        """Return adaptive strength based on open position performance."""
        if side == "flat":
            self._pos_side = None
            self._entry_price = None
            return 0.0
        strength = 1.0
        if self._pos_side and self._entry_price:
            # positive when current position is in profit
            pnl = (price - self._entry_price) / self._entry_price
            if self._pos_side == "sell":
                pnl = -pnl
            if side == self._pos_side:
                strength += pnl
            else:
                strength = -pnl
        if strength > 0:
            self._pos_side = side
            self._entry_price = price
        else:
            self._pos_side = None
            self._entry_price = None
        return strength

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        price_col = "close" if "close" in df.columns else "price"
        price = float(df[price_col].iloc[-1])
        if last_rsi > self.upper:
            strength = self._calc_strength("sell", price)
            return Signal("sell", strength)
        if last_rsi < self.lower:
            strength = self._calc_strength("buy", price)
            return Signal("buy", strength)
        strength = self._calc_strength("flat", price)
        return Signal("flat", strength)


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate mean reversion signals for backtesting."""

    df = data.copy()
    window = params.get("window", 14)
    threshold = params.get("threshold", 0.0)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)
    sl_pct = params.get("stop_loss", 0.0)
    tp_pct = params.get("take_profit", 0.0)

    ma = df["price"].rolling(window).mean()
    df["signal"] = 0
    df.loc[df["price"] < ma - threshold, "signal"] = 1
    df.loc[df["price"] > ma + threshold, "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["stop_loss"] = df["price"] * (1 - sl_pct)
    df["take_profit"] = df["price"] * (1 + tp_pct)
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "stop_loss", "take_profit", "fee", "slippage"]]
