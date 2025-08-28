# Supported venues

The following table lists the available venue identifiers, the market type they
target and the primary connection method used by each adapter.

| Venue | Market type | Connection method |
| ----- | ----------- | ----------------- |
| binance_spot | Spot | REST |
| binance_spot_ws | Spot | WebSocket |
| binance_futures | Futures (USD-M) | REST |
| binance_futures_ws | Futures (USD-M) | WebSocket |
| bybit_spot | Spot | REST |
| bybit_futures | Futures | REST |
| bybit_ws | Futures | WebSocket |
| okx_spot | Spot | REST |
| okx_futures | Futures | REST |
| okx_futures_ws | Futures | WebSocket |
| deribit | Perpetual futures | REST |
| deribit_ws | Perpetual futures | WebSocket |

Deribit ofrece perpetuos no solo de BTC y ETH sino también de varios altcoins
como SOL, XRP, MATIC, DOT, ADA, DOGE, LTC y TRX.

## Exchange configuration

Per-venue settings such as maker/taker fees or tick sizes can be defined in
`config/config.yaml` under an ``exchange_configs`` section:

```yaml
exchange_configs:
  binance_spot:
    market_type: spot
    maker_fee_bps: 10.0
    taker_fee_bps: 10.0
    tick_size: 0.01
```

Venues without an explicit entry fall back to automatic ``_spot``/``_perp``
inference for the market type.

