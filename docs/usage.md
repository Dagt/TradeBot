# Uso de TradingBot

Este proyecto incluye una interfaz de línea de comandos y una API para
controlar estrategias.  A continuación algunos ejemplos rápidos.

## CLI

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
python -m tradingbot.cli run-bot --symbol BTC/USDT
python -m tradingbot.cli backtest data/btcusdt_1m.csv --symbol BTC/USDT --strategy breakout_atr
python -m tradingbot.cli report --venue binance_spot_testnet
python -m tradingbot.cli tri-arb --route BTC-ETH-USDT --notional 200
python -m tradingbot.cli cross-arb --symbol BTC/USDT --spot binance_spot --perp bybit_futures
```

El comando ``tri-arb`` monitorea una ruta triangular especificada en formato
``BASE-MID-QUOTE`` y ejecuta las tres patas en modo *paper* con el notional
indicado.  ``cross-arb`` permite arbitrar diferencias entre un mercado spot y
uno perp dado un par de adaptadores.

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

## Panel y monitoreo

El repositorio incluye un `docker-compose.yml` que levanta la API junto con
TimescaleDB, Prometheus y Grafana para visualización.

```bash
docker compose up
```

Una vez levantados los servicios:

* API y dashboard estático: <http://localhost:8000> (`/metrics` expone métricas de Prometheus)
* Prometheus: <http://localhost:9090>
* Grafana: <http://localhost:3000> (usuario `admin`, contraseña `admin`)

