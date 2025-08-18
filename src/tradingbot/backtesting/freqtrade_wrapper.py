"""Minimal utilities to run Freqtrade strategies on in-memory data.

The wrapper provides helpers to load OHLCV data from CSV files and to
execute a Freqtrade ``IStrategy`` on a pandas ``DataFrame``.  Only a very
small subset of Freqtrade's functionality is supported – it merely
applies the strategy's entry/exit signals and simulates a long-only
portfolio.  The dependency on ``freqtrade`` is optional; an informative
``RuntimeError`` is raised if it is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Type, Any

import pandas as pd

__all__ = ["load_ohlcv", "run_strategy"]


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from ``path`` into a DataFrame.

    The CSV file must contain at least the columns ``timestamp``, ``open``,
    ``high``, ``low``, ``close`` and ``volume``.  The timestamp is parsed as
    UTC and set as the index.
    """

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df.set_index("timestamp", inplace=True)
    return df


def run_strategy(
    ohlcv: pd.DataFrame,
    strategy_cls: Type[Any],
    *,
    stake_amount: float = 100.0,
) -> Dict[str, Any]:
    """Execute ``strategy_cls`` on ``ohlcv`` and return basic metrics.

    Parameters
    ----------
    ohlcv:
        DataFrame with OHLCV columns.  The ``close`` price is used for
        executions.
    strategy_cls:
        Class implementing the Freqtrade strategy interface, i.e. providing
        ``populate_indicators``, ``populate_entry_trend`` and
        ``populate_exit_trend`` methods.
    stake_amount:
        Starting capital for the simple long-only simulation.

    Returns
    -------
    dict
        Dictionary with the final ``equity`` and a list of executed
        ``trades``.  Each trade contains the entry and exit prices along with
        the profit in currency units.
    """

    try:  # pragma: no cover - optional dependency
        import freqtrade  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("freqtrade is not installed") from exc

    strategy = strategy_cls()
    metadata = {"pair": "TEST/USDT"}
    df = ohlcv.copy()
    df = strategy.populate_indicators(df, metadata)
    df = strategy.populate_entry_trend(df, metadata)
    df = strategy.populate_exit_trend(df, metadata)

    if {"enter_long", "exit_long"} <= set(df.columns):
        entry_col, exit_col = "enter_long", "exit_long"
    elif {"buy", "sell"} <= set(df.columns):
        entry_col, exit_col = "buy", "sell"
    else:
        raise RuntimeError(
            "Strategy must define 'enter_long'/'exit_long' or 'buy'/'sell' columns"
        )

    entries = df[entry_col].fillna(False).astype(bool).tolist()
    exits = df[exit_col].fillna(False).astype(bool).tolist()
    prices = df["close"].tolist()

    equity = stake_amount
    position_price = None
    trades = []

    for price, entry, exit_ in zip(prices, entries, exits):
        if position_price is None and entry:
            position_price = price
        elif position_price is not None and exit_:
            pnl = (price / position_price - 1.0) * equity
            equity += pnl
            trades.append({"entry": position_price, "exit": price, "pnl": pnl})
            position_price = None

    return {"equity": equity, "trades": trades}

