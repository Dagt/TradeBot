"""Helpers for position sizing based on volatility targets.

This module also provides utilities to translate signal strength into actual
position deltas via ``notional = equity * strength``, ensuring consistent
pyramiding and reduction across strategies. Local stop‑losses use ``risk_pct``
as ``notional * risk_pct``.
"""

from __future__ import annotations


def vol_target(atr: float, equity: float, vol_target: float) -> float:
    """Return target position size given a volatility estimate.

    Parameters
    ----------
    atr:
        Average true range or volatility estimate of the asset.
    equity:
        Current account equity.
    vol_target:
        Fraction of equity to risk on a move of ``atr``. For example,
        ``0.02`` risks 2% of equity if price moves by one ATR.

    Returns
    -------
    float
        Desired absolute position size.  If any argument is non-positive,
        ``0.0`` is returned.

    Examples
    --------
    >>> vol_target(atr=2.0, equity=10.0, vol_target=1.0)
    5.0
    """
    if atr <= 0 or vol_target <= 0 or equity <= 0:
        return 0.0

    budget = equity * vol_target
    return budget / atr


def delta_from_strength(
    strength: float,
    equity: float,
    price: float,
    current_qty: float,
    risk_pct: float | None = None,
) -> float:
    """Translate a signal ``strength`` into a position delta.

    ``strength`` controls notional through ``notional = equity * strength``.
    Values above ``1.0`` pyramid exposure, values between ``0`` and ``1`` scale
    it down and negatives close or flip the position. When ``risk_pct`` is
    provided, the resulting position is capped at
    ``(equity * risk_pct) / price`` before calculating the delta. The target
    quantity is ``(equity * strength) / price`` and the delta from
    ``current_qty`` is returned.

    Parameters
    ----------
    strength:
        Target exposure as a fraction of account equity. Positive values denote
        long positions, negative values short. Values outside ``[-1, 1]`` are
        allowed and pyramid exposure proportionally.
    equity:
        Current account equity.
    price:
        Asset price used to convert notional into quantity.
    current_qty:
        Existing position size.
    risk_pct:
        Optional fraction of equity to risk. When supplied and positive, it
        limits the target quantity to ``±(equity * risk_pct) / price``.

    Returns
    -------
    float
        Signed quantity required to reach the target exposure from
        ``current_qty``.

    Examples
    --------
    >>> delta_from_strength(0.5, 10_000, 100, 20)
    30.0
    >>> delta_from_strength(0.2, 10_000, 100, 30)
    -10.0
    >>> delta_from_strength(1.5, 10_000, 100, 0)
    150.0
    >>> delta_from_strength(-0.5, 10_000, 100, 40)
    -90.0
    """

    if price <= 0:
        return -current_qty

    target_qty = (equity * strength) / price

    if risk_pct is not None and risk_pct > 0 and equity > 0:
        max_pos = (equity * risk_pct) / price
        if target_qty > max_pos:
            target_qty = max_pos
        elif target_qty < -max_pos:
            target_qty = -max_pos

    return target_qty - current_qty

