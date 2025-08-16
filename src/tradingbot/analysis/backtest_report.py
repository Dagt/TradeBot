"""Utilities to compute simple statistics from backtest results."""

from __future__ import annotations

from typing import Dict, List

import math
from statistics import NormalDist

import pandas as pd


def _stress_tests(returns: pd.Series, initial: float) -> Dict[str, float]:
    """Apply simple percentage shocks to each return and report final equity.

    The shocks are expressed as absolute percentage drops applied to every
    period.  For instance a ``-0.05`` shock represents a uniform 5% drop on all
    returns.
    """

    scenarios = {"drop_5": -0.05, "drop_10": -0.10}
    results: Dict[str, float] = {}
    for name, shock in scenarios.items():
        stressed = (1 + returns + shock).cumprod() * initial
        results[name] = float(stressed.iloc[-1])
    return results

def generate_report(result: Dict) -> Dict[str, float]:
    """Generate a basic backtest report.

    Parameters
    ----------
    result: dict returned by :func:`EventDrivenBacktestEngine.run`.

    Returns
    -------
    Mapping containing ``pnl`` (final equity), ``fill_rate`` and average
    ``slippage`` per traded unit.  If an ``equity_curve`` is present in the
    result, additional statistics ``sharpe``, ``sortino`` and
    ``deflated_sharpe_ratio`` are also returned.
    """

    equity = float(result.get("equity", 0.0))
    orders: List[Dict] = result.get("orders", [])  # type: ignore[assignment]

    total_qty = sum(o.get("qty", 0.0) for o in orders)
    total_filled = sum(o.get("filled", 0.0) for o in orders)
    fill_rate = total_filled / total_qty if total_qty else 0.0

    latencies = [o.get("latency") for o in orders if o.get("latency") is not None]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    total_slip = 0.0
    for o in orders:
        filled = o.get("filled", 0.0)
        if filled <= 0:
            continue
        if o.get("side") == "buy":
            slip = (o.get("avg_price", 0.0) - o.get("place_price", 0.0)) * filled
        else:
            slip = (o.get("place_price", 0.0) - o.get("avg_price", 0.0)) * filled
        total_slip += slip
    avg_slippage = total_slip / total_filled if total_filled else 0.0

    stats = {
        "pnl": equity,
        "fill_rate": fill_rate,
        "slippage": avg_slippage,
        "avg_latency": avg_latency,
    }

    eq_curve = result.get("equity_curve")
    if eq_curve and len(eq_curve) > 1:
        # Accept a list of numbers or dicts with an "equity" key
        if isinstance(eq_curve[0], dict):
            curve = [float(x.get("equity", 0.0)) for x in eq_curve]
        else:
            curve = [float(x) for x in eq_curve]
        returns = pd.Series(curve).pct_change().dropna()
        if not returns.empty and returns.std(ddof=0) > 0:
            daily_sharpe = returns.mean() / returns.std(ddof=0)
            sharpe = float(daily_sharpe * math.sqrt(252))

            downside = returns[returns < 0].std(ddof=0)
            sortino = (
                float(returns.mean() / downside * math.sqrt(252))
                if downside and downside > 0
                else 0.0
            )

            n = returns.shape[0]
            skew = float(returns.skew())
            kurt = float(returns.kurtosis())
            num = daily_sharpe * math.sqrt(n - 1)
            den = math.sqrt(max(1e-12, 1 - skew * daily_sharpe + (kurt - 1) / 4 * daily_sharpe**2))
            dsr = NormalDist().cdf(num / den)

            stats.update(
                {
                    "sharpe": sharpe,
                    "sortino": sortino,
                    "deflated_sharpe_ratio": float(dsr),
                    "stress_tests": _stress_tests(returns, curve[0]),
                }
            )

    return stats


__all__ = ["generate_report"]
