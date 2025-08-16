"""Simple wrapper to select the storage backend at runtime.

The application can persist market data either in TimescaleDB or QuestDB.  The
choice is controlled by ``settings.db_backend``.  This module exposes a uniform
API so callers can simply import ``tradingbot.storage`` and obtain the
appropriate helpers.
"""

from ..config import settings

if settings.db_backend.lower() == "questdb":
    from .quest import get_engine, insert_trade, insert_orderbook  # noqa: F401
else:  # default to timescale
    from .timescale import get_engine, insert_trade, insert_orderbook  # noqa: F401

__all__ = ["get_engine", "insert_trade", "insert_orderbook"]

