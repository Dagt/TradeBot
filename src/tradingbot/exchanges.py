"""Exchange configuration shared across modules."""

from __future__ import annotations

SUPPORTED_EXCHANGES: dict[str, dict] = {
    "binance": {"ccxt": "binance"},
    "okx": {"ccxt": "okx", "options": {"options": {"defaultType": "spot"}}},
    "bybit": {"ccxt": "bybit", "options": {"options": {"defaultType": "spot"}}},
}

__all__ = ["SUPPORTED_EXCHANGES"]
