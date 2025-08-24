"""Exchange configuration shared across modules."""

from __future__ import annotations

SUPPORTED_EXCHANGES: dict[str, dict] = {
    "binance_spot": {"ccxt": "binance", "options": {"defaultType": "spot"}},
    "binance_futures": {"ccxt": "binanceusdm", "options": {"defaultType": "future"}},
    "okx_spot": {"ccxt": "okx", "options": {"defaultType": "spot"}},
    "okx_futures": {"ccxt": "okx", "options": {"defaultType": "swap"}},
    "bybit_spot": {"ccxt": "bybit", "options": {"defaultType": "spot"}},
    "bybit_futures": {"ccxt": "bybit", "options": {"defaultType": "swap"}},
    "deribit_futures": {"ccxt": "deribit"},
}

__all__ = ["SUPPORTED_EXCHANGES"]
