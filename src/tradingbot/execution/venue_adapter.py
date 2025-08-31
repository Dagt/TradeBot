"""Helpers for translating generic order flags into venue specific parameters.

The :func:`translate_order_flags` function accepts a venue name together with
common order execution flags and returns a dictionary of parameters ready to be
forwarded to CCXT's ``create_order`` (or equivalent) method.  This keeps the
translation logic in a single place so individual connectors remain small.

Currently the translator understands the following flags:

``post_only``
    Whether the order should only add liquidity.  For venues that use a special
    ``timeInForce`` value (``GTX`` on Binance) the function handles the mapping.
``time_in_force``
    Optional TIF policy such as ``IOC`` or ``FOK``.  If ``post_only`` is also
    requested the maker flag takes precedence.
``iceberg_qty``
    Quantity for iceberg orders.  The parameter name differs per exchange.
``reduce_only``
    Ensure an order only decreases an existing position.  Typically maps to a
    ``reduceOnly`` parameter on derivative venues.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def translate_order_flags(
    venue: str,
    *,
    post_only: bool = False,
    time_in_force: Optional[str] = None,
    iceberg_qty: Optional[float] = None,
    reduce_only: bool = False,
) -> Dict[str, Any]:
    """Translate generic order flags into venue specific ``params``.

    Parameters
    ----------
    venue:
        Name of the exchange/venue.  A prefix match is used so ``binance``,
        ``binance_futures`` and ``binance_spot`` all map to the same rules.
    post_only:
        Whether the order should be maker only.
    time_in_force:
        Desired TIF policy.  Ignored when ``post_only`` is ``True`` for venues
        that use the ``GTX`` encoding.
    iceberg_qty:
        Visible portion for iceberg orders.
    reduce_only:
        For derivative venues this flag ensures the order only reduces an
        existing position.  Many venues share the same ``reduceOnly`` parameter
        name.

    Returns
    -------
    dict
        Mapping of parameters to pass to the exchange specific client.
    """

    v = venue.lower()
    params: Dict[str, Any] = {}

    # Binance -------------------------------------------------------------
    if v.startswith("binance"):
        if post_only:
            params["timeInForce"] = "GTX"
        elif time_in_force:
            params["timeInForce"] = time_in_force
        if iceberg_qty is not None:
            params["icebergQty"] = float(iceberg_qty)
        if reduce_only:
            params["reduceOnly"] = True
        return params

    # Bybit / OKX ---------------------------------------------------------
    if v.startswith("bybit") or v.startswith("okx"):
        if time_in_force:
            params["timeInForce"] = time_in_force
        if post_only:
            params["postOnly"] = True
        if iceberg_qty is not None:
            params["iceberg"] = float(iceberg_qty)
        if reduce_only:
            params["reduceOnly"] = True
        return params

    # Fallback for other venues ------------------------------------------
    if time_in_force:
        params["timeInForce"] = time_in_force
    if post_only:
        params["postOnly"] = True
    if iceberg_qty is not None:
        params["icebergQty"] = float(iceberg_qty)
    if reduce_only:
        params["reduceOnly"] = True
    return params


__all__ = ["translate_order_flags"]
