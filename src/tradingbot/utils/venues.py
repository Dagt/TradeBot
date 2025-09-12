from __future__ import annotations

def is_spot(venue: str) -> bool:
    """Return True if ``venue`` denotes a spot market.

    A venue is considered spot when its identifier ends with ``"_spot"`` or
    contains ``"_spot_"`` (e.g. ``"binance_spot_testnet"``).  This small helper
    mirrors the backtester logic without modifying it so both live runners and
    the backtest remain aligned.
    """
    v = venue.lower()
    return v.endswith("_spot") or "_spot_" in v
