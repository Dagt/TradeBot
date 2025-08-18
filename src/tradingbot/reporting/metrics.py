from __future__ import annotations

"""Portfolio performance metrics utilities.

This module provides small, dependency-light functions to compute
common performance statistics from an equity curve or a series of
returns.  All functions return ``0.0`` when the metric cannot be
computed due to insufficient data.
"""

from typing import Sequence, Dict

import math
import pandas as pd


def _to_returns(equity_curve: Sequence[float]) -> pd.Series:
    """Convert an equity curve to periodic returns."""

    series = pd.Series(list(equity_curve), dtype="float64")
    return series.pct_change().dropna()


def sharpe_ratio(returns: Sequence[float], periods: int = 252) -> float:
    """Compute annualised Sharpe ratio from periodic returns."""

    r = pd.Series(list(returns), dtype="float64")
    if r.empty or r.std(ddof=0) == 0:
        return 0.0
    return float((r.mean() / r.std(ddof=0)) * math.sqrt(periods))


def sortino_ratio(returns: Sequence[float], periods: int = 252) -> float:
    """Compute annualised Sortino ratio."""

    r = pd.Series(list(returns), dtype="float64")
    downside = r[r < 0]
    if r.empty or downside.std(ddof=0) == 0:
        return 0.0
    return float((r.mean() / downside.std(ddof=0)) * math.sqrt(periods))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Return the maximum drawdown of an equity curve."""

    if not equity_curve:
        return 0.0
    series = pd.Series(list(equity_curve), dtype="float64")
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calmar_ratio(equity_curve: Sequence[float], periods: int = 252) -> float:
    """Compute Calmar ratio using annualised CAGR over max drawdown."""

    returns = _to_returns(equity_curve)
    if returns.empty:
        return 0.0
    cagr = (1 + returns).prod() ** (periods / len(returns)) - 1
    mdd = abs(max_drawdown(equity_curve))
    return float(cagr / mdd) if mdd > 0 else 0.0


def hit_rate(returns: Sequence[float]) -> float:
    """Percentage of periods with positive return."""

    r = pd.Series(list(returns), dtype="float64")
    if r.empty:
        return 0.0
    return float((r > 0).sum() / len(r))


def evaluate(equity_curve: Sequence[float]) -> Dict[str, float]:
    """Compute a set of basic metrics from an equity curve."""

    returns = _to_returns(equity_curve)
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(equity_curve),
        "hit_rate": hit_rate(returns),
        "max_drawdown": max_drawdown(equity_curve),
    }


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "hit_rate",
    "max_drawdown",
    "evaluate",
]
