# TradingBot MVP (Local)

MVP para un bot de trading intradía, scalping y arbitraje en cripto (extensible a otros mercados).

## Estructura

```
tradingbot-mvp/
├─ requirements.txt
├─ pyproject.toml
├─ .env.example
├─ docker-compose.yml
├─ README.md
├─ sql/
│  └─ schema_timescale.sql
├─ src/tradingbot/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ logging_conf.py
│  ├─ bus.py
│  ├─ types.py
│  ├─ adapters/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ binance.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ ingestion.py
│  │  └─ features.py
│  ├─ storage/
│  │  ├─ __init__.py
│  │  ├─ timescale.py
│  │  └─ quest.py
│  ├─ strategies/
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  └─ breakout_atr.py
│  ├─ risk/
│  │  ├─ __init__.py
│  │  └─ manager.py
│  ├─ execution/
│  │  ├─ __init__.py
│  │  ├─ order_types.py
│  │  └─ router.py
│  ├─ backtest/
│  │  ├─ __init__.py
│  │  └─ event_engine.py
│  ├─ apps/
│  │  ├─ __init__.py
│  │  └─ api/
│  │     ├─ __init__.py
│  │     └─ main.py
│  └─ cli/
│     ├─ __init__.py
│     └─ main.py
   └─ tests/
   └─ test_smoke.py
```

## Documentación

La carpeta `docs/` contiene material adicional:

- [Arquitectura](docs/architecture.md)
- [Guía de setup](docs/setup.md)
- [Ejemplos de uso](docs/examples.md)

## Setup local

```bash
# Crear y activar entorno virtual (Python 3.11+)
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# Copiar variables de entorno
cp .env.example .env  # editar con tus credenciales

# Ejecutar pruebas para verificar la instalación
pytest
```

## Despliegue con Docker

Para levantar toda la pila de servicios (API, bases de datos, monitoreo):

```bash
docker compose up -d
```

También puedes utilizar los scripts de ayuda en `bin/`:

```bash
./bin/start_timescale.sh   # sólo TimescaleDB
./bin/start_questdb.sh     # sólo QuestDB
./bin/start_stack.sh       # stack completo
```

Si solo necesitas las bases de datos puedes utilizar los archivos en `sql/`:

```bash
# TimescaleDB
docker compose -f sql/docker-compose.timescale.yml up -d

# QuestDB
docker compose -f sql/docker-compose.questdb.yml up -d
```

## Variables de entorno

Copia `.env.example` a `.env` y completa las claves según corresponda. Variables principales:

- `BINANCE_API_KEY`, `BINANCE_API_SECRET`: credenciales de Binance Spot.
- `BYBIT_API_KEY`, `BYBIT_API_SECRET`: credenciales de Bybit Spot.
- `BINANCE_FUTURES_API_KEY`, `BINANCE_FUTURES_API_SECRET`, `BINANCE_FUTURES_TESTNET`, `BINANCE_FUTURES_LEVERAGE`: acceso a futuros.
- `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DB`: conexión a la base de datos.
- `ENV`, `LOG_LEVEL`: parámetros de ejecución.
- `SENTRY_DSN`: opcional para reportar errores a Sentry.

Consulta [docs/security.md](docs/security.md) para buenas prácticas de gestión de claves API.

## Límites de consulta por exchange

| Exchange | Límite aproximado | Notas |
|----------|------------------|-------|
| Binance Spot | 1200 weight/min (≈20 req/s) | REST público |
| Binance Futures | 1200 weight/min (≈20 req/s) | USDⓈ-M |
| Bybit | 50 req/s público, 10 req/s privado | compartir entre endpoints |
| OKX | 60 req/2s | límite global REST |

## Uso rápido

```bash
# 1) Crear entorno (recomendado python 3.11+)
python -m venv .venv && . .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# 2) Levantar TimescaleDB (opcional pero recomendado)
docker compose up -d timescaledb

# 3) Configurar variables
cp .env.example .env
# edita .env con tus API keys (retiros desactivados, claves restringidas)
```

### Claves API sólo-trade
Genera credenciales con los permisos mínimos:

1. Crea una nueva clave en el panel de la exchange.
2. Activa únicamente permisos de **lectura** y **trade**.
3. Deshabilita retiros y, si es posible, restringe por IP.
4. Guarda `API_KEY` y `API_SECRET` en `~/.secrets` (formato `CLAVE=valor`) o en el archivo `.env`.

```bash
# 4) Probar backtest de ejemplo (simulado, CSV)
python -m tradingbot.cli backtest --data ./data/examples/btcusdt_1m.csv

# 4b) Estrategia Order Flow (CSV con `bid_qty` y `ask_qty`)
python -m tradingbot.cli backtest --data /path/to/orderbook.csv --strategy order_flow

# 5) Iniciar API de control/monitoreo
uvicorn tradingbot.apps.api.main:app --reload --port 8080
```

## Configuración desde YAML con Hydra

Puedes definir configuraciones complejas en un archivo YAML y ejecutarlas con Hydra.

```yaml
# data/examples/backtest.yaml
csv_paths:
  BTC/USDT: data/examples/btcusdt_1m.csv
strategies:
  - [breakout_atr, BTC/USDT]
latency: 1
window: 120
mlflow:
  run_name: example_backtest
```

Ejecuta el backtest:

```bash
python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml
```

## Registro de resultados con MLflow

Si la configuración YAML incluye la sección `mlflow`, los resultados del backtest
se registrarán automáticamente en un experimento de MLflow (equity final y número de fills).

## Optimización de hiperparámetros con Optuna

```python
from tradingbot.backtest.event_engine import optimize_strategy_optuna

study = optimize_strategy_optuna(
    csv_path="data/examples/btcusdt_1m.csv",
    symbol="BTC/USDT",
    strategy_name="breakout_atr",
    param_space={
        "ema_n": {"type": "int", "low": 10, "high": 40},
        "atr_n": {"type": "int", "low": 5, "high": 30},
        "mult": {"type": "float", "low": 1.0, "high": 3.0},
    },
    n_trials=20,
)
print("Mejores parámetros:", study.best_params)
```

> **Nota**: este repo es un esqueleto funcional. Los adaptadores WS/REST y ejecución están stubs listos para ser implementados paso a paso.

## Estrategias

### Order Flow

Estrategia basada en el indicador *Order Flow Imbalance* (OFI). Calcula el
promedio del OFI en una ventana configurable y genera señales de compra o venta
cuando supera los umbrales definidos (`buy_threshold`, `sell_threshold`).

Ejemplo de uso con el CLI:

```bash
python -m tradingbot.cli backtest --data /path/to/orderbook.csv --strategy order_flow
```

El CSV debe contener las columnas `bid_qty` y `ask_qty`.

### Ingesta de Open Interest

Los adaptadores de exchanges (`Binance`, `Bybit`, `OKX`, etc.) exponen el método
`fetch_oi` para consultar el *open interest*. Para almacenar periódicamente este
dato en TimescaleDB se puede lanzar la tarea `poll_open_interest` para cada
exchange/símbolo:

```python
import asyncio
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.data.ingestion import poll_open_interest

adapter = BinanceFuturesAdapter(api_key, api_secret)
asyncio.create_task(poll_open_interest(adapter, "BTC/USDT"))

# Repetir con otros adaptadores: BybitFuturesAdapter, OKXFuturesAdapter, etc.
```

## Esquema de datos y carga

El archivo `sql/schema_timescale.sql` crea el esquema `market` con tablas básicas para el almacenamiento de mercado:

- `trades`: ejecuciones individuales
- `orderbook`: snapshots del libro de órdenes (`bid_px`, `bid_qty`, `ask_px`, `ask_qty`)
- `bars`: agregados OHLCV
- `funding`: tasas de funding de perps
- `open_interest`: interés abierto por exchange/símbolo

Para cargar la estructura en una instancia de TimescaleDB:

```bash
psql -h localhost -U postgres -f sql/schema_timescale.sql
```

Luego puedes insertar datos desde Python utilizando los helpers:

```python
from tradingbot.storage.timescale import get_engine, insert_funding, insert_orderbook

engine = get_engine()
insert_funding(engine, ts=..., exchange="binance", symbol="BTCUSDT", rate=0.0001, interval_sec=3600)
insert_orderbook(
    engine,
    ts=..., exchange="binance", symbol="BTCUSDT",
    bid_px=[...], bid_qty=[...], ask_px=[...], ask_qty=[...],
)
```

## Monitoreo y alertas

Este proyecto expone métricas Prometheus para latencia de APIs, errores de websockets, fills de órdenes, slippage y eventos de riesgo. Los dashboards de ejemplo para Grafana se encuentran en `monitoring/grafana`.

Para reportar excepciones a Sentry, define `SENTRY_DSN` en tu archivo `.env`. La configuración de logging inicializará Sentry automáticamente cuando este valor esté presente.

## CI/CD

El flujo de integración continua se ejecuta con GitHub Actions mediante `.github/workflows/ci.yml`. En cada `push` o `pull_request` se instala Python 3.11, las dependencias del proyecto y se ejecutan las pruebas con `pytest`.

## Scripts de inicio rápido

Algunos comandos útiles para comenzar a experimentar con el proyecto:

```bash
python -m tradingbot.cli backtest --data ./data/examples/btcusdt_1m.csv
python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml
uvicorn tradingbot.apps.api.main:app --reload --port 8080
```
