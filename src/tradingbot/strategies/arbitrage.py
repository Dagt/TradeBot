"""Simple spread based arbitrage strategy.

This module exposes a minimal ``Strategy`` subclass so the arbitrage
implementation becomes available in the dashboard's strategy selector.  The
runtime strategy is intentionally simplistic – it merely checks the spread
between two assets and never issues live orders – but it allows operators to
experiment and extend it further.
"""

from typing import Any

import pandas as pd

from .base import Strategy, Signal


PARAM_INFO: dict[str, str] = {}


class Arbitrage(Strategy):
    """Placeholder arbitrage strategy used for live experimentation."""

    name = "arbitrage"

    def __init__(self, **_: Any) -> None:
        """Accept arbitrary parameters for future extensions."""

        # No configuration required for the placeholder implementation.

    def on_bar(self, bar: dict[str, Any]) -> Signal | None:  # pragma: no cover -
        """Inspect ``bar`` data and return ``Signal`` if conditions met.

        The current implementation does not produce trading signals. It serves
        as a stub so the arbitrage strategy appears in the UI.  Users can
        replace this with a real implementation later on.
        """

        return Signal("flat", 0.0)


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate arbitrage signals for backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Data with ``asset_a`` and ``asset_b`` price columns.
    params : dict
        Parameters including ``threshold``, ``position_size``, ``fee`` y
        ``slippage``.

    Returns
    -------
    pd.DataFrame
        Data with generated signals and risk management levels.
    """

    df = data.copy()
    threshold = params.get("threshold", 0.0)
    position_size = params.get("position_size", 1)
    fee = params.get("fee", 0.0)
    slippage = params.get("slippage", 0.0)
    sl_pct = params.get("stop_loss", 0.0)
    tp_pct = params.get("take_profit", 0.0)

    spread = df["asset_a"] - df["asset_b"]
    df["signal"] = 0
    df.loc[spread > threshold, "signal"] = -1
    df.loc[spread < -threshold, "signal"] = 1

    df["position"] = df["signal"].shift(1).fillna(0) * position_size
    df["stop_loss"] = df["asset_a"] * (1 - sl_pct)
    df["take_profit"] = df["asset_a"] * (1 + tp_pct)
    df["fee"] = df["position"].abs() * fee
    df["slippage"] = df["position"].abs() * slippage

    return df[["signal", "position", "stop_loss", "take_profit", "fee", "slippage"]]


__all__ = ["Arbitrage", "generate_signals"]

