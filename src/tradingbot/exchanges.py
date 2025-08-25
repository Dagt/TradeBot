"""Exchange configuration shared across modules."""

from __future__ import annotations

SUPPORTED_EXCHANGES: dict[str, dict] = {
    "binance_spot": {"ccxt": "binance", "defaultType": "spot"},
    "binance_futures": {"ccxt": "binanceusdm", "defaultType": "future"},
    "okx_spot": {"ccxt": "okx", "defaultType": "spot"},
    "okx_futures": {"ccxt": "okx", "defaultType": "swap"},
    "bybit_spot": {"ccxt": "bybit", "defaultType": "spot"},
    "bybit_futures": {"ccxt": "bybit", "defaultType": "swap"},
    "deribit_futures": {"ccxt": "deribit", "defaultType": "future"},
}

__all__ = ["SUPPORTED_EXCHANGES"]
