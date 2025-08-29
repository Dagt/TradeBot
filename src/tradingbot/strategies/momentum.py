import pandas as pd
from .base import StatefulStrategy, Signal, record_signal_metrics
from ..data.features import rsi, calc_ofi

class Momentum(StatefulStrategy):
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
        rsi_n: int = 14,
        rsi_threshold: float = 60.0,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        super().__init__(tp_pct=tp_pct, sl_pct=sl_pct, max_hold_bars=max_hold_bars)
        self.rsi_n = rsi_n
        self.threshold = rsi_threshold

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Signal | None:
        df: pd.DataFrame = bar["window"]
        if len(df) < self.rsi_n + 1:
            return None
        price = float(df["close"].iloc[-1]) if "close" in df.columns else None
        if price is not None:
            exit_sig = self.check_exit(price)
            if exit_sig:
                return exit_sig
        rsi_series = rsi(df, self.rsi_n)
        last_rsi = rsi_series.iloc[-1]
        ofi_val = 0.0
        if {"bid_qty", "ask_qty"}.issubset(df.columns):
            ofi_val = calc_ofi(df[["bid_qty", "ask_qty"]]).iloc[-1]
        if self.pos_side is None:
            if last_rsi > self.threshold and ofi_val >= 0 and price is not None:
                return self.open_position("buy", price)
            if last_rsi < 100 - self.threshold and ofi_val <= 0 and price is not None:
                return self.open_position("sell", price)
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
