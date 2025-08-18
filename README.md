# TradeBot

TradeBot es un bot de trading modular orientado a estrategias intradía,
scalping y arbitraje sobre criptomonedas.  Incluye todo lo necesario para
ingerir datos, realizar backtesting y ejecutar estrategias en modo
``paper`` o real desde una interfaz web.

## Características principales

- Ingesta de datos en tiempo real (WebSocket y REST) para Binance, Bybit,
  OKX y otros exchanges.
- Motor de estrategias con ejemplos de momentum, mean reversion, arbitraje
  triangular y arbitraje spot/perp.
- Gestión de riesgo y portafolio con límites de exposición y ``kill switch``.
- Router de ejecución con algoritmos TWAP/VWAP/POV y soporte maker/taker.
- Backtester vectorizado y motor event‑driven con modelado de slippage.
- **Panel web** con métricas en vivo y un **ejecutor de comandos CLI** que
  permite lanzar cualquier comando desde el navegador.

## Requisitos

- Python 3.11+
- ``git``
- ``docker`` (opcional, para bases de datos y monitoreo)

## Instalación rápida

```bash
git clone <repo>
cd TradeBot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completa con tus claves
```

## Ejecución del panel web

El panel expone métricas, PnL y una consola para ejecutar comandos de la
CLI.

```bash
uvicorn tradingbot.apps.api.main:app --reload --port 8000
```

Visita `http://localhost:8000` e inicia sesión con las credenciales
definidas en `API_USER`/`API_PASS` (por defecto `admin`/`admin`).

### Consola de comandos

En la sección **Comandos CLI** del panel puedes ejecutar cualquier comando
de `tradingbot.cli`.  Ejemplos:

1. Escribe `backtest-cfg data/examples/backtest.yaml` y pulsa **Ejecutar**.
2. Usa `tri-arb BTC-ETH-USDT --notional 50` para disparar un arbitraje
   triangular de prueba.

La salida de `stdout` y `stderr` aparecerá debajo del formulario.

## Uso desde la línea de comandos

La CLI está basada en [Typer](https://typer.tiangolo.com/) y ofrece
subcomandos para las distintas tareas del proyecto.

```bash
python -m tradingbot.cli --help

# Ingesta de libro de órdenes
python -m tradingbot.cli ingest BTC/USDT --depth 20

# Descarga histórica desde Kaiko
python -m tradingbot.cli ingest-historical kaiko BTC/USDT --kind trades

# Backtest a partir de un YAML de configuración
python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml

# Arbitraje triangular
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 50
```

Todos estos comandos pueden ejecutarse también desde el panel web gracias a
la nueva sección de **Comandos CLI**.

## Ejecutar pruebas

El proyecto incluye una batería extensa de tests.  En entornos con recursos
limitados puede ejecutarse una versión reducida:

```bash
PYTHONPATH=src:. pytest tests/test_smoke.py
```

## Estructura del proyecto

```
TradeBot/
├─ src/tradingbot/          # Código fuente del bot
├─ monitoring/              # Métricas y paneles de observabilidad
├─ docs/                    # Documentación adicional
├─ tests/                   # Pruebas unitarias e integración
└─ data/, sql/, bin/, etc.  # Scripts y ejemplos
```

---

Este repositorio es una base extensible.  Añade tus propias estrategias y
conectores según sea necesario y utilízalo tanto para uso personal como para
compartirlo con terceros.

