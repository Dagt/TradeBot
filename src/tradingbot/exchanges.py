"""Exchange configuration shared across modules."""

from __future__ import annotations

SUPPORTED_EXCHANGES: dict[str, dict] = {
    "binance_spot": {"ccxt": "binance"},
    "binance_futures": {"ccxt": "binanceusdm"},
    "okx_spot": {"ccxt": "okx", "options": {"options": {"defaultType": "spot"}}},
    "okx_futures": {"ccxt": "okx", "options": {"options": {"defaultType": "swap"}}},
    "bybit_spot": {"ccxt": "bybit", "options": {"options": {"defaultType": "spot"}}},
    "bybit_futures": {"ccxt": "bybit", "options": {"options": {"defaultType": "swap"}}},
    "deribit_futures": {"ccxt": "deribit"},
}

__all__ = ["SUPPORTED_EXCHANGES"]
