"""Thin wrapper around vectorbt to perform parameter sweeps.

The wrapper accepts OHLCV data and a strategy ``signal`` function.  It
evaluates the strategy for every combination in ``param_grid`` using
``vectorbt`` and returns a DataFrame with basic performance metrics.

The dependency on ``vectorbt`` is optional; an informative ``RuntimeError`` is
raised if it is not installed.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple, Any

import pandas as pd

__all__ = ["run_parameter_sweep"]


def run_parameter_sweep(
    ohlcv: pd.DataFrame,
    signal_func: Callable[..., Tuple[pd.Series, pd.Series]],
    param_grid: Dict[str, Iterable],
    *,
    price_col: str = "close",
    portfolio_kwargs: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a vectorbt backtest over all parameter combinations.

    Parameters
    ----------
    ohlcv:
        DataFrame containing at least the ``price_col`` column.
    signal_func:
        Callable that receives the price Series and parameters, returning
        entry and exit boolean Series.
    param_grid:
        Mapping of parameter names to iterables of values to sweep.
    price_col:
        Column in ``ohlcv`` to use as the price series. Defaults to ``close``.
    portfolio_kwargs:
        Additional kwargs forwarded to ``vectorbt.Portfolio.from_signals``.

    Returns
    -------
    pandas.DataFrame
        Metrics ``sharpe_ratio``, ``max_drawdown`` and ``total_return`` indexed
        by the parameter combinations.
    """

    try:  # pragma: no cover - optional dependency
        import vectorbt as vbt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("vectorbt is not installed") from exc

    close = ohlcv[price_col]
    param_names = list(param_grid.keys())

    def apply_func(close, *args):
        params = dict(zip(param_names, args))
        return signal_func(close, **params)

    factory = vbt.IndicatorFactory(
        input_names=["close"],
        param_names=param_names,
        output_names=["entries", "exits"],
    )
    indicator = factory.from_apply_func(apply_func)
    ind = indicator.run(close, **param_grid)

    freq = None
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        freq = pd.infer_freq(ohlcv.index)
    if freq is None:
        freq = "1D"

    pf_kwargs = dict(portfolio_kwargs or {})
    pf = vbt.Portfolio.from_signals(
        close, ind.entries, ind.exits, freq=freq, **pf_kwargs
    )
    metrics = pd.concat(
        [pf.sharpe_ratio(), pf.max_drawdown(), pf.total_return()], axis=1
    )
    metrics.columns = ["sharpe_ratio", "max_drawdown", "total_return"]
    metrics.index.set_names(list(param_grid.keys()), inplace=True)
    return metrics

