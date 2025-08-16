"""Backtesting helpers built on top of vectorbt.

This module exposes :func:`run_vectorbt` which evaluates a strategy for
multiple parameter combinations using `vectorbt`_.  The strategy class must
expose a static ``signal`` method that receives the close price series along
with strategy parameters and returns entry and exit boolean series.

Example
-------
>>> import numpy as np, pandas as pd, vectorbt as vbt
>>> from tradingbot.backtest.vectorbt_engine import run_vectorbt
>>> class MAStrategy:
...     @staticmethod
...     def signal(close, fast, slow):
...         fast_ma = vbt.MA.run(close, fast)
...         slow_ma = vbt.MA.run(close, slow)
...         entries = fast_ma.ma_crossed_above(slow_ma)
...         exits = fast_ma.ma_crossed_below(slow_ma)
...         return entries, exits
>>> price = pd.Series(np.linspace(1, 2, 100))
>>> data = pd.DataFrame({'close': price})
>>> run_vectorbt(data, MAStrategy, {'fast': [5, 10], 'slow': [20]})
"""

from __future__ import annotations

from typing import Dict, Iterable, Type

import pandas as pd
import vectorbt as vbt


def run_vectorbt(
    data: pd.DataFrame, strategy_cls: Type, params: Dict[str, Iterable]
) -> pd.DataFrame:
    """Run a vectorbt backtest over a grid of parameters.

    Parameters
    ----------
    data:
        Price data containing a ``close`` column.
    strategy_cls:
        Strategy class exposing a static ``signal`` method that returns
        entry and exit Series compatible with :func:`vectorbt.Portfolio.from_signals`.
    params:
        Mapping of parameter name to an iterable of values.  All combinations are
        evaluated.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by parameter combinations with columns:
        ``sharpe_ratio``, ``max_drawdown`` and ``total_return``.
    """

    close = data["close"]
    factory = vbt.IndicatorFactory(
        input_names=["close"],
        param_names=list(params.keys()),
        output_names=["entries", "exits"],
    )
    indicator = factory.from_apply_func(strategy_cls.signal)
    ind = indicator.run(close, **params)

    # Try to infer frequency for annualised metrics
    freq = None
    if isinstance(data.index, pd.DatetimeIndex):
        freq = pd.infer_freq(data.index)
    if freq is None:
        freq = "1D"
    pf = vbt.Portfolio.from_signals(close, ind.entries, ind.exits, freq=freq)

    metrics = pd.concat(
        [pf.sharpe_ratio(), pf.max_drawdown(), pf.total_return()], axis=1
    )
    metrics.columns = ["sharpe_ratio", "max_drawdown", "total_return"]
    metrics.index.set_names(list(params.keys()), inplace=True)
    return metrics
