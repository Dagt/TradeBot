# Uso de TradingBot

Este proyecto incluye una interfaz de línea de comandos y una API para
controlar estrategias.  A continuación algunos ejemplos rápidos.

## CLI

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
python -m tradingbot.cli run-bot --symbol BTC/USDT
python -m tradingbot.cli backtest data/btcusdt_1m.csv --symbol BTC/USDT --strategy breakout_atr
python -m tradingbot.cli backtest-cfg src/tradingbot/config/config.yaml
python -m tradingbot.cli report --venue binance_spot_testnet
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 100
python -m tradingbot.cli cross-arb BTC/USDT binance_spot binance_futures --threshold 0.001 --notional 50
```

`tri-arb` lanza un pequeño lazo de arbitraje triangular en Binance. La ruta se
especifica como ``BASE-MID-QUOTE`` (p.ej. ``BTC-ETH-USDT``) y ``--notional``
define el capital en la divisa ``quote``.

`cross-arb` busca oportunidades de arbitraje entre un mercado spot y otro
perpetuo utilizando los adapters indicados. El umbral de premium se controla con
``--threshold`` y ``--notional`` establece el tamaño por pata.

### Configuraciones con Hydra

El comando `backtest-cfg` permite cargar la configuración desde un archivo
YAML utilizando `hydra`.  La estructura básica es la siguiente:

```yaml
exchanges:
  binance_api_key: ""
  binance_api_secret: ""

strategies:
  default: breakout_atr
  params: {}

backtest:
  data: data/examples/btcusdt_1m.csv
  symbol: BTC/USDT
  strategy: breakout_atr
  latency: 1
  window: 120

storage:
  backend: sqlite
  url: sqlite:///:memory:
```

Ejemplo de ejecución:

```bash
python -m tradingbot.cli backtest-cfg src/tradingbot/config/config.yaml
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


La estrategia Cash & Carry genera señales ``buy`` cuando el funding es positivo y la base supera el umbral indicado, o ``sell`` cuando el funding es negativo y la base cae por debajo de dicho umbral.

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

