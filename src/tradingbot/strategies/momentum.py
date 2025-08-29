import pandas as pd
from .base import Strategy, Signal, record_signal_metrics
from ..data.features import rsi

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

    def __init__(
        self,
        *,
        tp_bps: float = 30.0,
        sl_bps: float = 40.0,
        max_hold_bars: int = 20,
        **kwargs,
    ):
        self.rsi_n = kwargs.get("rsi_n", 14)
        self.threshold = kwargs.get("rsi_threshold", 55.0)
        self.tp_bps = float(tp_bps)
        self.sl_bps = float(sl_bps)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: int = 0
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        price_col = "close" if "close" in df.columns else "price"
        price = float(df[price_col].iloc[-1]) if price_col in df.columns else None
        buy = last_rsi > self.threshold
        sell = last_rsi < 100 - self.threshold

        if self.pos_side == 0:
            if buy:
                self.pos_side = 1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("buy", 1.0)
            if sell:
                self.pos_side = -1
                self.entry_price = price
                self.hold_bars = 0
                return Signal("sell", 1.0)
            return None

        self.hold_bars += 1
        exit_signal = (buy and self.pos_side < 0) or (sell and self.pos_side > 0)
        exit_tp = exit_sl = False
        if price is not None and self.entry_price is not None:
            pnl_bps = (
                (price - self.entry_price) / self.entry_price * 10000 * self.pos_side
            )
            exit_tp = pnl_bps >= self.tp_bps
            exit_sl = pnl_bps <= -self.sl_bps
        exit_time = self.hold_bars >= self.max_hold_bars
        if exit_signal or exit_tp or exit_sl or exit_time:
            side = "sell" if self.pos_side > 0 else "buy"
            self.pos_side = 0
            self.entry_price = None
            self.hold_bars = 0
            return Signal(side, 1.0)
        return None


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate momentum signals for backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Price data with a ``price`` column.
    params : dict
        Parameters including ``window``, ``position_size``, ``stop_loss``,
        ``take_profit``, ``fee`` and ``slippage``.

    Returns
    -------
    pd.DataFrame
        Data with signal, position, stop-loss/take-profit levels and
        transaction cost estimates.
    """

    df = data.copy()
    window = params.get("window", 14)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)
    sl_pct = params.get("stop_loss", 0.0)
    tp_pct = params.get("take_profit", 0.0)

    ma = df["price"].rolling(window).mean()
    df["signal"] = 0
    df.loc[df["price"] > ma, "signal"] = 1
    df.loc[df["price"] < ma, "signal"] = -1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["stop_loss"] = df["price"] * (1 - sl_pct)
    df["take_profit"] = df["price"] * (1 + tp_pct)
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "stop_loss", "take_profit", "fee", "slippage"]]
