"""Simple spread based arbitrage strategy."""

import pandas as pd


def generate_signals(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate arbitrage signals for backtesting.

    Parameters
    ----------
    data : pd.DataFrame
        Data with ``asset_a`` and ``asset_b`` price columns.
    params : dict
        Parameters including ``threshold``, ``position_size``, ``stop_loss``,
        ``take_profit``, ``fee`` and ``slippage``.

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


__all__ = ["generate_signals"]

