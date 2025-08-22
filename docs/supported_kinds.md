# Supported stream kinds by exchange

The CLI exposes different market data streams depending on the selected
exchange adapter.  The table below lists the available ``kind`` values for
each venue as detected by :func:`get_supported_kinds`. Venue names follow the
pattern ``<exchange>_<market>`` for REST adapters and ``<exchange>_<market>_ws``
for WebSocket adapters; the suffix ``_testnet`` is appended automatically when
connecting to test environments.

| Exchange | Supported kinds |
| -------- | --------------- |
| binance_spot | trades, orderbook, bba, delta |
| binance_futures | trades, orderbook, bba, delta, funding, open_interest |
| binance_spot_ws | trades, trades_multi, orderbook, bba, delta |
| binance_futures_ws | trades, trades_multi, orderbook, bba, delta, funding, open_interest |
| bybit_spot | trades, orderbook, bba, delta |
| bybit_futures | trades, orderbook, bba, delta, funding, open_interest |
| bybit_futures_ws | trades, orderbook, bba, delta, funding, open_interest |
| okx_spot | trades, orderbook, bba, delta |
| okx_futures | trades, orderbook, bba, delta, funding, open_interest |
| okx_futures_ws | trades, orderbook, bba, delta, funding, open_interest |
| deribit_futures | trades, funding, open_interest |
| deribit_futures_ws | trades, orderbook, bba, delta |

This reference aims to keep the UI and CLI documentation aligned with the
actual capabilities of each adapter.

