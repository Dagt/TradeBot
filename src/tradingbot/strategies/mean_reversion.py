import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi


PARAM_INFO = {
    "rsi_n": "Ventana para el cálculo del RSI",
    "upper": "Nivel RSI superior para vender",
    "lower": "Nivel RSI inferior para comprar",
    "tp_bps": "Take profit en puntos básicos",
    "sl_bps": "Stop loss en puntos básicos",
    "max_hold_bars": "Barras máximas en posición",
    "scale_by": "Método para escalar la fuerza de la señal",
}

class MeanReversion(Strategy):
    """RSI based mean reversion strategy with adaptive strength.

    Besides generating ``buy``/``sell`` signals based on RSI levels,
    the strategy adjusts the ``strength`` of those signals according to the
    performance of the current position or the distance of RSI from the
    threshold levels. The same tracking logic used in
    :class:`ScalpPingPong` is applied to handle take profit, stop loss and
    maximum holding time.

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
    tp_bps : float, optional
        Take profit in basis points, by default ``30``.
    sl_bps : float, optional
        Stop loss in basis points, by default ``40``.
    max_hold_bars : int, optional
        Maximum number of bars to hold a trade, by default ``15``.
    scale_by : {"pnl", "rsi"}, optional
        Method used to scale signal ``strength``, by default ``"pnl"``.
    """

    name = "mean_reversion"

    def __init__(self, **kwargs):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.upper = kwargs.get("upper", 60.0)
        self.lower = kwargs.get("lower", 40.0)
        self.tp_bps = kwargs.get("tp_bps", 30.0)
        self.sl_bps = kwargs.get("sl_bps", 40.0)
        self.max_hold_bars = kwargs.get("max_hold_bars", 15)
        self.scale_by = kwargs.get("scale_by", "pnl")
        # Track current position to adapt strength
        self._pos_side: str | None = None
        self._entry_price: float | None = None
        self._hold_bars: int = 0

    def _calc_strength(self, side: str, price: float, last_rsi: float) -> float:
        """Return adaptive strength based on PnL or RSI distance."""
        strength = 1.0
        if self.scale_by == "pnl" and self._pos_side and self._entry_price:
            # positive when current position is in profit
            pnl = (price - self._entry_price) / self._entry_price
            if self._pos_side == "sell":
                pnl = -pnl
            if side == self._pos_side:
                strength += pnl
            else:
                strength = -pnl
        elif self.scale_by == "rsi":
            if side == "buy":
                strength = min(1.0, (self.lower - last_rsi) / self.lower)
            else:
                strength = min(1.0, (last_rsi - self.upper) / (100 - self.upper))
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

        if self._pos_side is not None:
            # manage existing position before evaluating new entries
            self._hold_bars += 1
            assert self._entry_price is not None
            pnl_bps = (price - self._entry_price) / self._entry_price * 10000
            if self._pos_side == "sell":
                pnl_bps = -pnl_bps
            exit_rsi = self.lower < last_rsi < self.upper
            exit_tp = pnl_bps >= self.tp_bps
            exit_sl = pnl_bps <= -self.sl_bps
            exit_time = self._hold_bars >= self.max_hold_bars
            if exit_rsi or exit_tp or exit_sl or exit_time:
                side = "sell" if self._pos_side == "buy" else "buy"
                self._pos_side = None
                self._entry_price = None
                self._hold_bars = 0
                return Signal(side, 1.0)

            if self._pos_side == "buy" and last_rsi < self.lower:
                strength = self._calc_strength("buy", price, last_rsi)
                return Signal("buy", strength)
            if self._pos_side == "sell" and last_rsi > self.upper:
                strength = self._calc_strength("sell", price, last_rsi)
                return Signal("sell", strength)
            return None

        if last_rsi > self.upper:
            strength = self._calc_strength("sell", price, last_rsi)
            self._pos_side = "sell"
            self._entry_price = price
            self._hold_bars = 0
            return Signal("sell", strength)
        if last_rsi < self.lower:
            strength = self._calc_strength("buy", price, last_rsi)
            self._pos_side = "buy"
            self._entry_price = price
            self._hold_bars = 0
            return Signal("buy", strength)
        return None


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
