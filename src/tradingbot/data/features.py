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


def _to_dataframe_lists(data: DataLike, columns: Sequence[str]) -> pd.DataFrame:
    """Normalise ``data`` into a DataFrame preserving list/array values.

    Unlike :func:`_to_dataframe`, this helper keeps list-like values untouched
    which is required when operating on L2 order book snapshots where each
    field contains multiple levels.
    """

    if isinstance(data, pd.DataFrame):
        return data.loc[:, columns]

    rows = []
    for snap in data:
        row = {col: snap.get(col) for col in columns}
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


def atr_ewma(data: DataLike, n: int = 14) -> pd.Series:
    """Average True Range using an exponential moving average.

    This variant applies an exponentially weighted moving average (EWMA)
    to the sequence of true ranges, mirroring the original formulation by
    Welles Wilder.  It offers a smoother estimate of volatility compared to
    the simple moving average used in :func:`atr`.

    Parameters
    ----------
    data:
        Source data containing ``high``, ``low`` and ``close`` columns.
    n:
        Lookback window for the exponential mean of true ranges.

    Returns
    -------
    pandas.Series
        The EWMA-based ATR values.
    """

    df = _to_dataframe(data, ["high", "low", "close"])
    high, low, close = df["high"], df["low"], df["close"]
    tr = np.maximum(
        high - low,
        np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()),
    )
    return tr.ewm(alpha=1 / n, adjust=False).mean()


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


def order_flow_imbalance(data: DataLike, depth: int = 1) -> pd.Series:
    """Order Flow Imbalance.

    When ``data`` contains Level 2 snapshots with ``bid_px``/``ask_px`` and
    ``bid_qty``/``ask_qty`` arrays, the imbalance is computed for the top
    ``depth`` levels using the formulation by Cartea et al.  For simple inputs
    exposing only ``bid_qty``/``ask_qty`` scalars, the function falls back to a
    top-of-book approximation identical to previous versions.

    Parameters
    ----------
    data:
        Input exposing order book fields.
    depth:
        Number of levels to include in the calculation.  Defaults to ``1``
        (top of book).

    Returns
    -------
    pandas.Series
        Sequence of OFI values where positive numbers indicate buying
        pressure.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")

    if isinstance(data, pd.DataFrame):
        has_depth = {"bid_px", "ask_px", "bid_qty", "ask_qty"}.issubset(
            data.columns
        )
    else:
        data = list(data)
        has_depth = bool(data) and all(
            k in data[0] for k in ("bid_px", "ask_px", "bid_qty", "ask_qty")
        )

    if not has_depth:
        # Fallback: treat inputs as top-of-book quantities
        df = _to_dataframe(data, ["bid_qty", "ask_qty"])
        bid = df["bid_qty"].to_numpy(np.float64)
        ask = df["ask_qty"].to_numpy(np.float64)
        ofi = _ofi_jit(bid, ask)
        return pd.Series(ofi, index=df.index, name="ofi")

    df = _to_dataframe_lists(data, ["bid_px", "bid_qty", "ask_px", "ask_qty"])
    bid_px = np.stack(df["bid_px"].to_numpy())[:, :depth]
    bid_qty = np.stack(df["bid_qty"].to_numpy())[:, :depth]
    ask_px = np.stack(df["ask_px"].to_numpy())[:, :depth]
    ask_qty = np.stack(df["ask_qty"].to_numpy())[:, :depth]

    out = np.zeros(len(df))
    for i in range(1, len(df)):
        val = 0.0
        for lvl in range(depth):
            bp, bp_prev = bid_px[i, lvl], bid_px[i - 1, lvl]
            bq, bq_prev = bid_qty[i, lvl], bid_qty[i - 1, lvl]
            ap, ap_prev = ask_px[i, lvl], ask_px[i - 1, lvl]
            aq, aq_prev = ask_qty[i, lvl], ask_qty[i - 1, lvl]

            if bp > bp_prev:
                val += bq
            elif bp < bp_prev:
                val -= bq_prev
            else:
                val += bq - bq_prev

            if ap < ap_prev:
                val -= aq
            elif ap > ap_prev:
                val += aq_prev
            else:
                val -= aq - aq_prev
        out[i] = val
    return pd.Series(out, index=df.index, name="ofi")


def depth_imbalance(data: DataLike, depth: int = 1) -> pd.Series:
    """Depth imbalance between bid and ask queues.

    Parameters
    ----------
    data:
        Input exposing ``bid_qty`` and ``ask_qty`` fields.
    depth:
        Number of levels to aggregate when quantities are provided as lists.
        Defaults to ``1`` (top of book).

    Returns
    -------
    pandas.Series
        Values in the ``[-1, 1]`` range where positive numbers signal more
        depth on the bid side.
    """

    if depth < 1:
        raise ValueError("depth must be at least 1")

    df = _to_dataframe_lists(data, ["bid_qty", "ask_qty"])

    first = df["bid_qty"].iloc[0]
    if isinstance(first, (list, tuple, np.ndarray, pd.Series)):
        bid_mat = np.stack(df["bid_qty"].to_numpy())[:, :depth]
        ask_mat = np.stack(df["ask_qty"].to_numpy())[:, :depth]
        bid = bid_mat.sum(axis=1)
        ask = ask_mat.sum(axis=1)
    else:
        bid = df["bid_qty"].to_numpy(np.float64)
        ask = df["ask_qty"].to_numpy(np.float64)

    di = _depth_jit(bid.astype(np.float64), ask.astype(np.float64))
    return pd.Series(di, index=df.index, name="depth_imbalance")


def book_vacuum(data: DataLike, threshold: float = 0.5) -> pd.Series:
    """Detect sudden drops in book depth ("vaciados").

    The detector compares bid and ask queue sizes with the previous snapshot
    and flags events where one side loses at least ``threshold`` proportion of
    its volume.  Positive values indicate that the ask side was wiped (bullish)
    while negative values denote a bid side wipe (bearish).  Zero is returned
    when no event is detected or for the first observation.

    Parameters
    ----------
    data:
        Iterable of snapshots or DataFrame with ``bid_qty`` and ``ask_qty``.
    threshold:
        Proportional drop required to trigger an event.  ``0.5`` means a
        reduction of 50% or more compared with the previous snapshot.
    """

    df = _to_dataframe_lists(data, ["bid_qty", "ask_qty"])
    bid = df["bid_qty"].to_numpy()
    ask = df["ask_qty"].to_numpy()

    if isinstance(bid[0], (list, tuple, np.ndarray, pd.Series)):
        bid = np.stack(bid)[:, 0]
        ask = np.stack(ask)[:, 0]

    out = np.zeros(len(df))
    for i in range(1, len(df)):
        b_prev, a_prev = bid[i - 1], ask[i - 1]
        b, a = bid[i], ask[i]
        if b_prev > 0 and (b_prev - b) / b_prev >= threshold:
            out[i] = -1
        elif a_prev > 0 and (a_prev - a) / a_prev >= threshold:
            out[i] = 1
    return pd.Series(out, index=df.index, name="book_vacuum")


def liquidity_gap(data: DataLike, threshold: float) -> pd.Series:
    """Detect large gaps between the first and second level of the book.

    Parameters
    ----------
    data:
        Input exposing ``bid_px`` and ``ask_px`` lists (at least two levels).
    threshold:
        Minimum absolute price gap to signal an event.

    Returns
    -------
    pandas.Series
        +1 for gaps on the ask side (bullish), ``-1`` for bid side gaps and ``0``
        otherwise.
    """

    df = _to_dataframe_lists(data, ["bid_px", "ask_px"])
    bid_px = df["bid_px"].to_numpy()
    ask_px = df["ask_px"].to_numpy()

    out = np.zeros(len(df))
    for i in range(len(df)):
        bp = bid_px[i]
        ap = ask_px[i]
        bid_gap = np.nan
        ask_gap = np.nan
        if isinstance(bp, (list, tuple, np.ndarray, pd.Series)) and len(bp) > 1:
            bid_gap = bp[0] - bp[1]
        if isinstance(ap, (list, tuple, np.ndarray, pd.Series)) and len(ap) > 1:
            ask_gap = ap[1] - ap[0]
        if bid_gap > threshold:
            out[i] = -1
        elif ask_gap > threshold:
            out[i] = 1
    return pd.Series(out, index=df.index, name="liquidity_gap")


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
    "atr_ewma",
    "rsi",
    "order_flow_imbalance",
    "depth_imbalance",
    "book_vacuum",
    "liquidity_gap",
    "returns",
    "keltner_channels",
]

