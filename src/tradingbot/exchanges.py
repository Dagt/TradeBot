"""Exchange configuration shared across modules."""

from __future__ import annotations

SUPPORTED_EXCHANGES: dict[str, dict] = {
    "binance": {"ccxt": "binance"},
    "binance_futures": {"ccxt": "binanceusdm"},
    "okx": {"ccxt": "okx", "options": {"defaultType": "spot"}},
    "okx_futures": {"ccxt": "okx", "options": {"defaultType": "swap"}},
    "bybit": {"ccxt": "bybit", "options": {"defaultType": "spot"}},
    "bybit_futures": {"ccxt": "bybit", "options": {"defaultType": "swap"}},
    "deribit": {"ccxt": "deribit"},
}

__all__ = ["SUPPORTED_EXCHANGES"]
