# Mapa del blueprint

Este documento enlaza los puntos del [blueprint inicial](../blueprint_trading_bot.md) con los módulos que los implementan dentro del repositorio. También se listan los comandos de la CLI asociados a cada capa cuando existen.

| Punto del blueprint | Módulos / archivos relevantes | Comandos de CLI |
|---------------------|-------------------------------|-----------------|
| **1. Ingesta de datos** | `adapters/`, `data/ingestion.py`, `workers/` | `python -m tradingbot.cli ingest`, `python -m tradingbot.cli ingest-historical`, `python -m tradingbot.cli ingestion-workers` |
| **2. Feature engineering** | `data/features.py`, `data/basis.py`, `data/open_interest.py` | – |
| **3. Señal / estrategia** | `strategies/`, `live/runner*.py` | `run-bot`, `paper-run`, `tri-arb`, `cross-arb`, `run-cross-arb` |
| **4. Gestión de cartera y riesgo** | `risk/manager.py`, `risk/portfolio_guard.py`, `risk/daily_guard.py` | `run-bot`, `daemon` |
| **5. Ejecución** | `execution/router.py`, `execution/algos.py`, `execution/order_types.py` | `run-bot --algo`, `run-cross-arb` |
| **6. Persistencia** | `storage/timescale.py`, `storage/quest.py` | `ingest`, `ingest-historical`, `ingestion-workers`, `report` |
| **7. Backtesting & simulación** | `backtest/event_engine.py`, `backtesting/engine.py`, `backtesting/vectorbt_wrapper.py` | `backtest`, `backtest-cfg`, `walk-forward`, `paper-run` |
| **8. Monitoreo & Ops** | `monitoring/panel.py`, `monitoring/metrics.py`, `utils/metrics.py` | `report` |

## Funcionalidades extra

Además de los puntos anteriores, el proyecto incorpora estrategias de
arbitraje avanzado, señales de microestructura, adaptadores para
testnets, un panel web con ejecución de comandos y una API de control
remoto.  La lista completa se describe en
[extra_features.md](extra_features.md).
