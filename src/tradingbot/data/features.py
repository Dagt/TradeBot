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


def atr(data: DataLike, n: int = 14) -> pd.Series:
    """Average True Range.

    ``data`` must provide ``high``, ``low`` and ``close`` fields.
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
    original indicator.  Values are bounded between 0 and 100.
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
    """Compute Order Flow Imbalance (OFI) using top of book changes.

    ``data`` must expose ``bid_qty`` and ``ask_qty`` fields representing the
    top of book queue sizes.
    """

    df = _to_dataframe(data, ["bid_qty", "ask_qty"])
    bid_delta = df["bid_qty"].diff().fillna(0)
    ask_delta = df["ask_qty"].diff().fillna(0)
    return bid_delta - ask_delta


def depth_imbalance(data: DataLike) -> pd.Series:
    """Depth imbalance between bid and ask queues.

    Defined as ``(bid_qty - ask_qty) / (bid_qty + ask_qty)``.  Values are
    bounded between -1 and 1 where positive numbers indicate a larger bid queue
    (buy side) and negative numbers indicate a larger ask queue (sell side).
    """

    df = _to_dataframe(data, ["bid_qty", "ask_qty"])
    bid_qty = df["bid_qty"]
    ask_qty = df["ask_qty"]
    denom = bid_qty + ask_qty
    denom = denom.replace(0, np.nan)  # avoid division by zero
    return (bid_qty - ask_qty) / denom


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
    "keltner_channels",
]

