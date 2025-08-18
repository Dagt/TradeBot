"""Utilities for normalising market symbols across exchanges.

The project interacts with multiple venues which may use different symbol
notations, e.g. ``BTCUSDT`` (Binance), ``BTC-USDT`` (Bybit) or
``BTC-USD-SWAP`` (OKX coin-margined perpetuals).  Internally we prefer a
compact representation without separators between the base and quote assets
and preserving any contract suffix after a single dash.  This module exposes a
simple :func:`normalize` helper implementing this mapping.

Examples
--------
>>> normalize("BTC/USDT")
'BTCUSDT'
>>> normalize("btc-usdt")
'BTCUSDT'
>>> normalize("BTC-USD-SWAP")
'BTCUSD-SWAP'

The function is intentionally lightweight and does not attempt to validate that
the provided symbol actually exists on a given exchange; it merely reshapes the
string into the internal format used across the codebase.
"""
from __future__ import annotations

__all__ = ["normalize"]


def normalize(symbol: str) -> str:
    """Return the internal representation for ``symbol``.

    Parameters
    ----------
    symbol: str
        Market symbol possibly using ``/`` or ``-`` as separators and an
        optional contract suffix (e.g. ``-SWAP``).

    Returns
    -------
    str
        Normalised symbol in the form ``BASEQUOTE`` or ``BASEQUOTE-SUFFIX``.
    """
    if not symbol:
        return symbol

    s = symbol.upper().replace("/", "-")
    parts: list[str] = [p for p in s.split("-") if p]
    if not parts:
        return s
    if len(parts) == 1:
        # Already normalised without separators
        return parts[0]

    base, quote, *rest = parts
    norm = base + quote
    if rest:
        norm += "-" + "-".join(rest)
    return norm
