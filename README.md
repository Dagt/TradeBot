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
│  │  ├─ ingest.py
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

# 4) Probar backtest de ejemplo (simulado, CSV)
python -m tradingbot.cli backtest --data ./data/examples/btcusdt_1m.csv

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

## Esquema de datos y carga

El archivo `sql/schema_timescale.sql` crea el esquema `market` con tablas básicas para el almacenamiento de mercado:

- `trades`: ejecuciones individuales
- `orderbook`: snapshots del libro de órdenes (`bid_px`, `bid_qty`, `ask_px`, `ask_qty`)
- `bars`: agregados OHLCV
- `funding`: tasas de funding de perps

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
