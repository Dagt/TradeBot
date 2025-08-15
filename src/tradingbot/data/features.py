import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(n).mean()

def keltner_channels(df: pd.DataFrame, ema_n: int = 20, atr_n: int = 14, mult: float = 1.5):
    ema = df['close'].ewm(span=ema_n, adjust=False).mean()
    _atr = atr(df, atr_n)
    upper = ema + mult * _atr
    lower = ema - mult * _atr
    return upper, lower


def rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Relative Strength Index of the ``close`` column.

    Uses an exponential moving average formulation which closely follows
    the original indicator.  Values are bounded between 0 and 100.
    """

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=n, adjust=False).mean()
    avg_loss = loss.ewm(span=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)


def order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """Compute Order Flow Imbalance (OFI) using top of book changes.

    The implementation follows a simplified formulation where OFI is the
    difference between changes in bid and ask queue sizes.  Positive values
    indicate buying pressure while negative values indicate selling pressure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``bid_qty`` and ``ask_qty`` representing the
        top of book quantities.

    Returns
    -------
    pd.Series
        Series of OFI values aligned with ``df``'s index.
    """

    bid_delta = df['bid_qty'].diff().fillna(0)
    ask_delta = df['ask_qty'].diff().fillna(0)
    return bid_delta - ask_delta


def depth_imbalance(df: pd.DataFrame) -> pd.Series:
    """Depth imbalance between bid and ask queues.

    Defined as ``(bid_qty - ask_qty) / (bid_qty + ask_qty)``.  Values are
    bounded between -1 and 1 where positive numbers indicate a larger bid
    queue (buy side) and negative numbers indicate a larger ask queue (sell
    side).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``bid_qty`` and ``ask_qty``.

    Returns
    -------
    pd.Series
        Depth imbalance series.
    """

    bid_qty = df['bid_qty']
    ask_qty = df['ask_qty']
    denom = bid_qty + ask_qty
    # Avoid division by zero
    denom = denom.replace(0, np.nan)
    return (bid_qty - ask_qty) / denom

