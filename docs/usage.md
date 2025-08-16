# Uso de TradingBot

Este proyecto incluye una interfaz de línea de comandos y una API para
controlar estrategias.  A continuación algunos ejemplos rápidos.

## CLI

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
python -m tradingbot.cli run-bot --symbol BTC/USDT
python -m tradingbot.cli backtest data/btcusdt_1m.csv --symbol BTC/USDT --strategy breakout_atr
python -m tradingbot.cli report --venue binance_spot_testnet
```

## Estrategias disponibles

Las estrategias pueden instanciarse pasando parámetros mediante ``**kwargs``.
Los nombres que deben utilizarse en la CLI (``--strategy``) o al usar la API
corresponden al atributo ``name`` de cada clase.

| Estrategia        | Nombre           | Parámetros principales                |
|-------------------|-----------------|--------------------------------------|
| Momentum          | ``momentum``    | ``rsi_n`` (int), ``rsi_threshold``    |
| Mean Reversion    | ``mean_reversion`` | ``rsi_n`` (int), ``upper`` y ``lower`` |
| Breakout por Vol. | ``breakout_vol``| ``lookback`` (int), ``mult`` (float)  |
| Triangular Arb.   | ``triangular_arb`` | ``taker_fee_bps``, ``buffer_bps``   |
| Cash & Carry      | ``cash_and_carry`` | ``threshold`` (float)               |


## API

```
POST /strategies/{name}/start
POST /strategies/{name}/stop
GET  /strategies/status
```

