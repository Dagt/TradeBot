"""Utilities to compute simple statistics from backtest results."""

from __future__ import annotations

from typing import Dict, List


def generate_report(result: Dict) -> Dict[str, float]:
    """Generate a basic backtest report.

    Parameters
    ----------
    result: dict returned by :func:`EventDrivenBacktestEngine.run`.

    Returns
    -------
    Mapping containing ``pnl`` (final equity), ``fill_rate`` and average
    ``slippage`` per traded unit.
    """

    equity = float(result.get("equity", 0.0))
    orders: List[Dict] = result.get("orders", [])  # type: ignore[assignment]

    total_qty = sum(o.get("qty", 0.0) for o in orders)
    total_filled = sum(o.get("filled", 0.0) for o in orders)
    fill_rate = total_filled / total_qty if total_qty else 0.0

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

    return {"pnl": equity, "fill_rate": fill_rate, "slippage": avg_slippage}


__all__ = ["generate_report"]
