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

> **Nota**: este repo es un esqueleto funcional. Los adaptadores WS/REST y ejecución están stubs listos para ser implementados paso a paso.
