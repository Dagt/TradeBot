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

from typing import Dict, Iterable, Type, Any, Tuple, List

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


def optimize_vectorbt_optuna(
    data: pd.DataFrame,
    strategy_cls: Type,
    param_space: Dict[str, Dict[str, Any]],
    n_trials: int = 20,
    metric: str = "sharpe_ratio",
) -> "optuna.study.Study":
    """Optimize strategy parameters using Optuna and vectorbt.

    The strategy's ``signal`` method is evaluated for parameter combinations
    suggested by Optuna.  The objective maximises the specified ``metric``
    (default Sharpe ratio).
    """

    import optuna  # type: ignore

    close = data["close"]

    def objective(trial: "optuna.trial.Trial") -> float:
        params: Dict[str, Any] = {}
        for name, space in param_space.items():
            t = space.get("type")
            low, high = space.get("low"), space.get("high")
            if t == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif t == "float":
                params[name] = trial.suggest_float(name, float(low), float(high))
            else:
                raise ValueError(f"unsupported param type: {t}")

        entries, exits = strategy_cls.signal(close, **params)
        freq = pd.infer_freq(data.index) or "1D"
        pf = vbt.Portfolio.from_signals(close, entries, exits, freq=freq)
        metric_val = getattr(pf, metric)()
        return float(metric_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def walk_forward(
    data: pd.DataFrame,
    strategy_cls: Type,
    params: Dict[str, Iterable],
    train_size: int,
    test_size: int,
) -> pd.DataFrame:
    """Perform walk-forward analysis using vectorbt.

    For each rolling window, parameters are selected on the training slice using
    :func:`run_vectorbt` and evaluated on the subsequent test slice.  Metrics
    include Sharpe ratio, drawdown, total return and Deflated Sharpe Ratio.
    """

    from ..utils.performance import dsr

    results: List[Dict[str, Any]] = []
    step = test_size
    for start in range(0, len(data) - train_size - test_size + 1, step):
        train = data.iloc[start : start + train_size]
        test = data.iloc[start + train_size : start + train_size + test_size]

        train_metrics = run_vectorbt(train, strategy_cls, params)
        best_idx = train_metrics["sharpe_ratio"].idxmax()
        if isinstance(best_idx, tuple):
            best_params = dict(zip(train_metrics.index.names, best_idx))
        else:
            best_params = {train_metrics.index.name or list(params.keys())[0]: best_idx}

        entries, exits = strategy_cls.signal(test["close"], **best_params)
        freq = pd.infer_freq(test.index) or "1D"
        pf = vbt.Portfolio.from_signals(test["close"], entries, exits, freq=freq)
        returns = pf.returns().values
        metrics = {
            "start": test.index[0] if isinstance(test.index, pd.Index) else start,
            "end": test.index[-1]
            if isinstance(test.index, pd.Index)
            else start + train_size + test_size,
            "sharpe_ratio": float(pf.sharpe_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
            "total_return": float(pf.total_return()),
            "dsr": dsr(returns, num_trials=len(train_metrics)),
        }
        metrics.update(best_params)
        results.append(metrics)

    return pd.DataFrame(results)
