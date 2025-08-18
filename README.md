# TradeBot

TradeBot es un bot de trading modular orientado a estrategias intradía,
scalping y arbitraje sobre criptomonedas.  Incluye todo lo necesario para
ingerir datos, realizar backtesting y ejecutar estrategias en modo
``paper`` o real desde una interfaz web.

La correspondencia entre el blueprint original y los módulos del código se documenta en [docs/blueprint_map.md](docs/blueprint_map.md).

### Explicación para principiantes

Un bot de trading es como un "piloto automático" que compra y vende
criptomonedas siguiendo un conjunto de reglas.  Tú defines las reglas y el
programa las ejecuta de forma rápida y sin emociones.  TradeBot ya trae
estrategias listas para usar y permite probarlas sin arriesgar dinero
gracias al **paper trading** (simulación).

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

## Exchanges y pares soportados

- **Exchanges**: [Binance](https://www.binance.com),
  [Bybit](https://www.bybit.com) y [OKX](https://www.okx.com).  El diseño es
  modular y pueden añadirse más.
- **Mercados**: pares spot y contratos perpetuos disponibles en esos
  exchanges.
- **Pares populares**: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT y cualquier
  otro listado por los exchanges anteriores.

## Estrategias incluidas

Cada estrategia se puede ejecutar en modo simulación o real.  A
continuación se explica la idea teórica y cómo la implementa TradeBot.

### Momentum intradía
**Idea**: cuando un precio sube con fuerza, suele seguir subiendo a corto
plazo.
**Implementación**: el bot calcula el retorno de los últimos minutos y
compra si supera un umbral; vende cuando el impulso se agota o cambia de
signo.

### Mean reversion
**Idea**: los precios tienden a volver a su promedio después de moverse
demasiado.
**Implementación**: se calcula una media móvil y su desviación.  Si el
precio está muy por encima, se vende; si está por debajo, se compra.

### Breakout de volatilidad
**Idea**: tras un periodo de calma, un movimiento brusco puede iniciar una
nuevo recorrido de precios.
**Implementación**: se observa la volatilidad (ATR) y se activan órdenes
cuando el precio rompe un canal predefinido.

### Arbitraje triangular
**Idea**: en un mismo exchange, las tasas de cambio entre tres pares pueden
quedar desalineadas.  Al hacer la ruta A→B→C→A se obtiene una ganancia
sin exposición direccional.
**Implementación**: el bot revisa continuamente rutas como BTC‑ETH‑USDT y
ejecuta las tres operaciones si el beneficio neto supera las comisiones.

### Arbitraje entre exchanges / cash‑and‑carry
**Idea**: un mismo activo puede tener precios distintos entre exchanges o
entre mercado spot y perp.  Comprar donde está barato y vender donde está
caro permite capturar la diferencia o el pago de funding.
**Implementación**: el bot compara precios de los exchanges conectados y
abre posiciones opuestas (spot vs perp o exchange vs exchange) cuando la
brecha supera un umbral configurado.

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
## Configuración inicial

1. Copia `.env.example` a `.env` y completa tus claves API (`BINANCE_KEY`,
   `BINANCE_SECRET`, etc.). Para pruebas en modo papel puedes dejar los
   valores vacíos.
2. (Opcional) Levanta la base de datos y el stack de monitoreo con Docker:

   ```bash
   docker-compose up -d
   ```

3. Inicia el panel web con métricas y consola de comandos:

   ```bash
   uvicorn monitoring.panel:app --reload --port 8000
   ```

   Luego visita `http://localhost:8000` en tu navegador.

## Comandos CLI

Todos los comandos están disponibles tanto desde la terminal como desde la
consola del panel web.

```bash
python -m tradingbot.cli <comando> [opciones]
```

Los runners (`run-bot`, `real-run`, `paper-run`) aceptan una opción
`--config` para cargar parámetros de la estrategia desde un archivo YAML.

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `ingest` | Stream de order book a la base de datos | `python -m tradingbot.cli ingest BTC/USDT --depth 20` |
| `ingest-historical` | Descarga histórica desde Kaiko o CoinAPI | `python -m tradingbot.cli ingest-historical kaiko BTC/USDT --kind trades` |
| `run-bot` | Ejecuta el bot en vivo o testnet | `python -m tradingbot.cli run-bot --exchange binance --symbol BTC/USDT --config cfg/estrategia.yaml` |
| `paper-run` | Ejecuta una estrategia en modo simulación | `python -m tradingbot.cli paper-run --symbol BTC/USDT --strategy breakout_atr --config cfg/estrategia.yaml` |
| `daemon` | Levanta el daemon de trading mediante Hydra | `python -m tradingbot.cli daemon config/config.yaml` |
| `ingestion-workers` | Workers de funding y open interest | `python -m tradingbot.cli ingestion-workers` |
| `backtest` | Backtest vectorizado desde CSV | `python -m tradingbot.cli backtest data/ohlcv.csv` |
| `backtest-cfg` | Backtest desde un YAML de configuración | `python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml` |
| `walk-forward` | Optimización walk-forward | `python -m tradingbot.cli walk-forward cfg/wf.yaml` |
| `report` | Resumen de PnL en TimescaleDB | `python -m tradingbot.cli report` |
| `train-ml` | Entrena una estrategia con ML | `python -m tradingbot.cli train-ml datos.csv target modelo.pkl` |
| `tri-arb` | Arbitraje triangular | `python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 50` |
| `cross-arb` | Arbitraje spot vs perp entre exchanges | `python -m tradingbot.cli cross-arb BTC/USDT binance_spot binance_futures` |
| `run-cross-arb` | Runner de arbitraje usando ExecutionRouter | `python -m tradingbot.cli run-cross-arb BTC/USDT binance_spot binance_futures` |

La salida de cada comando aparecerá tanto en la terminal como en la consola
del panel web.

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

