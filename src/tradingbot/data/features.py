"""Feature engineering utilities.

This module exposes a small collection of indicators that can operate on a
``pandas.DataFrame`` or on an iterable of Level 2 snapshots (dictionaries with
columns such as ``bid_qty``/``ask_qty`` or ``high``/``low``/``close``).  The
helpers normalise the input so that downstream functions can focus purely on
the numerical computations.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Any, Union

import numpy as np
import pandas as pd

try:  # pragma: no cover - numba is optional at runtime
    from numba import njit
except Exception:  # pragma: no cover - fall back to pure Python
    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        if args:
            return args[0]
        return wrapper


DataLike = Union[pd.DataFrame, Iterable[Mapping[str, Any]]]


def _to_dataframe(data: DataLike, columns: Sequence[str]) -> pd.DataFrame:
    """Normalise ``data`` into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    data:
        Either a DataFrame with the required ``columns`` or an iterable of
        mapping objects (e.g. L2 snapshots).  When lists/arrays are encountered
        in the mapping values, the first element is used which corresponds to
        the top of book.
    columns:
        Columns to extract from the input.
    """

    if isinstance(data, pd.DataFrame):
        return data.loc[:, columns]

    rows = []
    for snap in data:
        row = {}
        for col in columns:
            val = snap.get(col)
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                row[col] = val[0] if len(val) > 0 else np.nan
            else:
                row[col] = val
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


@njit
def _ofi_jit(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Numba implementation of Order Flow Imbalance."""

    out = np.empty(bid.size)
    out[0] = 0.0
    for i in range(1, bid.size):
        out[i] = (bid[i] - bid[i - 1]) - (ask[i] - ask[i - 1])
    return out


@njit
def _depth_jit(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    """Numba implementation of depth imbalance."""

    out = np.empty(bid.size)
    for i in range(bid.size):
        denom = bid[i] + ask[i]
        out[i] = (bid[i] - ask[i]) / denom if denom != 0 else np.nan
    return out


@njit
def _returns_jit(close: np.ndarray, log: bool) -> np.ndarray:
    """Numba implementation of returns calculation."""

    out = np.empty(close.size)
    out[0] = np.nan
    for i in range(1, close.size):
        if log:
            out[i] = np.log(close[i] / close[i - 1])
        else:
            out[i] = close[i] / close[i - 1] - 1.0
    return out


def atr(data: DataLike, n: int = 14) -> pd.Series:
    """Average True Range.

    Parameters
    ----------
    data:
        Source data containing ``high``, ``low`` and ``close`` columns.
    n:
        Lookback window for the rolling mean of true ranges.

    Returns
    -------
    pandas.Series
        The ATR values.
    """

    df = _to_dataframe(data, ["high", "low", "close"])
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()),
    )
    return tr.rolling(n).mean()


def rsi(data: DataLike, n: int = 14) -> pd.Series:
    """Relative Strength Index of the ``close`` column.

    Uses an exponential moving average formulation which closely follows the
    original indicator. Values are bounded between 0 and 100.

    Parameters
    ----------
    data:
        Source data containing a ``close`` column.
    n:
        Window used for the exponential averages.

    Returns
    -------
    pandas.Series
        RSI values.
    """

    df = _to_dataframe(data, ["close"])
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=n, adjust=False).mean()
    avg_loss = loss.ewm(span=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)


def order_flow_imbalance(data: DataLike) -> pd.Series:
    """Order Flow Imbalance based on top of book changes.

    Parameters
    ----------
    data:
        Input exposing ``bid_qty`` and ``ask_qty`` fields.

    Returns
    -------
    pandas.Series
        Sequence of OFI values where positive numbers indicate buying
        pressure.
    """

    df = _to_dataframe(data, ["bid_qty", "ask_qty"])
    bid = df["bid_qty"].to_numpy(np.float64)
    ask = df["ask_qty"].to_numpy(np.float64)
    ofi = _ofi_jit(bid, ask)
    return pd.Series(ofi, index=df.index, name="ofi")


def depth_imbalance(data: DataLike) -> pd.Series:
    """Depth imbalance between bid and ask queues.

    Parameters
    ----------
    data:
        Input exposing ``bid_qty`` and ``ask_qty`` fields.

    Returns
    -------
    pandas.Series
        Values in the ``[-1, 1]`` range where positive numbers signal more
        depth on the bid side.
    """

    df = _to_dataframe(data, ["bid_qty", "ask_qty"])
    bid = df["bid_qty"].to_numpy(np.float64)
    ask = df["ask_qty"].to_numpy(np.float64)
    di = _depth_jit(bid, ask)
    return pd.Series(di, index=df.index, name="depth_imbalance")


def returns(data: DataLike, log: bool = True) -> pd.Series:
    """Compute sequential returns from ``close`` prices.

    Parameters
    ----------
    data:
        Input containing a ``close`` column.
    log:
        When ``True`` compute log returns, otherwise simple percentage
        returns.

    Returns
    -------
    pandas.Series
        Series of returns with ``NaN`` as the first element.
    """

    df = _to_dataframe(data, ["close"])
    close = df["close"].to_numpy(np.float64)
    out = _returns_jit(close, log)
    return pd.Series(out, index=df.index, name="returns")


def keltner_channels(
    data: DataLike,
    ema_n: int = 20,
    atr_n: int = 14,
    mult: float = 1.5,
):
    """Keltner channel upper/lower bounds."""

    df = _to_dataframe(data, ["close", "high", "low"])
    ema = df["close"].ewm(span=ema_n, adjust=False).mean()
    _atr = atr(df, atr_n)
    upper = ema + mult * _atr
    lower = ema - mult * _atr
    return upper, lower


__all__ = [
    "atr",
    "rsi",
    "order_flow_imbalance",
    "depth_imbalance",
    "returns",
    "keltner_channels",
]

