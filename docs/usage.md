# Uso de TradingBot

Este proyecto incluye una interfaz de línea de comandos y una API para
controlar estrategias.  A continuación algunos ejemplos rápidos.

Las funcionalidades adicionales (arbitrajes, panel web con CLI, API de
control) se documentan en [extra_features.md](extra_features.md).

## CLI

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
python -m tradingbot.cli run-bot --symbol BTC/USDT
python -m tradingbot.cli backtest data/btcusdt_1m.csv --symbol BTC/USDT --strategy breakout_atr
python -m tradingbot.cli paper-run --strategy breakout_atr --symbol BTC/USDT
python -m tradingbot.cli backtest-cfg src/tradingbot/config/config.yaml
python -m tradingbot.cli report --venue binance_spot_testnet
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 100
python -m tradingbot.cli cross-arb BTC/USDT binance_spot binance_futures --threshold 0.001 --notional 50
python bin/download_history.py funding kaiko BTC-USD --exchange binance --backend csv
python bin/download_history.py open-interest coinapi BTCUSD --backend csv
```

Los nuevos comandos `funding` y `open-interest` de `bin/download_history.py`
permiten descargar tasas de funding y datos de open interest utilizando
conectores REST de Kaiko o CoinAPI y persistirlos en TimescaleDB, QuestDB o
archivos CSV.

`tri-arb` lanza un pequeño lazo de arbitraje triangular en Binance. La ruta se
especifica como ``BASE-MID-QUOTE`` (p.ej. ``BTC-ETH-USDT``) y ``--notional``
define el capital en la divisa ``quote``.

`cross-arb` busca oportunidades de arbitraje entre un mercado spot y otro
perpetuo utilizando los adapters indicados. El umbral de premium se controla con
``--threshold`` y ``--notional`` establece el tamaño por pata.

`paper-run` permite ejecutar cualquier estrategia en modo papel y expone las
métricas de Prometheus en `http://localhost:8000/metrics/prometheus`.

```bash
python -m tradingbot.cli paper-run --strategy breakout_atr --symbol BTC/USDT
```

El **daemon en vivo** ahora puede ejecutar este arbitraje cruzado de manera
automática entre Binance, Bybit y OKX.  El proceso concilia periódicamente los
balances por exchange y actualiza tanto el `RiskManager` como el
`PortfolioGuard` con las posiciones y PnL por venue.

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

* API y dashboard estático: <http://localhost:8000> (`/metrics` devuelve métricas agregadas, Prometheus en `/metrics/prometheus`)
* Prometheus: <http://localhost:9090>
* Grafana: <http://localhost:3000> (usuario `admin`, contraseña `admin`)

## Optimización walk-forward

El módulo de backtesting incluye una función de optimización **walk-forward**
que realiza una búsqueda en rejilla sobre distintos parámetros y evalúa cada
combinación en ventanas secuenciales de entrenamiento/prueba.  Se registran los
resultados de cada split en el log y la función retorna un listado con los
detalles:

```python
from tradingbot.backtesting.engine import walk_forward_optimize

param_grid = [
    {"rsi_n": 10, "rsi_threshold": 55},
    {"rsi_n": 14, "rsi_threshold": 60},
]

results = walk_forward_optimize(
    "data/examples/btcusdt_1m.csv",
    "BTC/USDT",
    "momentum",
    param_grid,
    train_size=1000,
    test_size=250,
)
print(results)
```

## MLflow y Optuna

El módulo `tradingbot.experiments` incluye helpers para registrar backtests en
**MLflow** y ejecutar optimizaciones genéricas con **Optuna**.

```python
from tradingbot.backtesting.engine import run_backtest_mlflow

res = run_backtest_mlflow(
    {"BTC/USDT": "data/examples/btcusdt_1m.csv"},
    [("breakout_atr", "BTC/USDT")],
    run_name="demo",
)
print(res["sharpe"])
```

Para búsquedas de hiperparámetros:

```python
from tradingbot.experiments import optimize
from tradingbot.backtesting.engine import run_backtest_csv

param_space = {
    "window": lambda t: t.suggest_int("window", 50, 150),
}

def backtest(params):
    return run_backtest_csv(
        {"BTC/USDT": "data/examples/btcusdt_1m.csv"},
        [("breakout_atr", "BTC/USDT")],
        window=params["window"],
    )

study = optimize(param_space, backtest, n_trials=5)
print(study.best_params)
```

## Interfaz mínima de monitoreo

`monitoring/panel.py` levanta una aplicación FastAPI que expone métricas
agregadas en `/metrics` (Prometheus en `/metrics/prometheus`) y el estado de las
estrategias en `/strategies/status`.  Al montar el directorio estático se incluye
un `index.html` sencillo que consulta ambos endpoints para mostrar un resumen
rápido.

