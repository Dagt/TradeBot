# TradeBot

TradeBot es un bot de trading modular orientado a estrategias intradía,
scalping y arbitraje sobre criptomonedas.  Incluye todo lo necesario para
ingerir datos, realizar backtesting y ejecutar estrategias en modo
``paper`` o real desde una interfaz web.

La correspondencia entre el blueprint original y los módulos del código se documenta en [docs/blueprint_map.md](docs/blueprint_map.md).

## Quickstart

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/<tu_usuario>/TradeBot.git
   cd TradeBot
   ```

2. **Crear `.env`**
   Copia el archivo `.env.example` a `.env` y completa las credenciales.
   ```bash
   cp .env.example .env
   ```

3. **Levantar los servicios**
   ```bash
   make up
   ```

4. **Ingerir datos**
   ```bash
   tradingbot ingest
   ```

5. **Ejecutar backtesting**
   ```bash
   tradingbot backtest
   ```

6. **Iniciar dashboards**
   ```bash
   uvicorn tradingbot.apps.api.main:app --reload --port 8000
   ```
   Visita `http://localhost:8000/` para gestionar credenciales,
   `http://localhost:8000/monitor` para monitoreo y
   `http://localhost:8000/bots` para lanzar bots o usar la CLI.

Al terminar, consulta [docs/blueprint_map.md](docs/blueprint_map.md) para entender la correspondencia entre el blueprint y el código.

## Errores comunes

| Error | Solución |
|-------|----------|
| `tradingbot: command not found` | Ejecuta los comandos dentro del contenedor o añade `src` al `PYTHONPATH`. |
| Falta el archivo `.env` | Crea uno con `cp .env.example .env` y completa las variables requeridas. |
| `make: *** No rule to make target 'up'` | Ejecuta `make` desde la raíz del proyecto donde está el `Makefile`. |
| `tradingbot backtest` sin datos | Ejecuta `tradingbot ingest` primero para descargar los datos necesarios. |

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
- **Panel web** dividido en secciones de credenciales, monitoreo y bots.
  La sección de bots incorpora un **ejecutor de comandos CLI** para
  lanzar cualquier comando desde el navegador y formularios para
  configurar estrategias sin usar la terminal.

## Funcionalidades extra

TradeBot incluye una serie de capacidades adicionales más allá del MVP
original. Entre ellas se destacan las estrategias de arbitraje
triangular y entre exchanges, señales basadas en microestructura,
adaptadores para múltiples venues con soporte de testnet, un panel web
que permite ejecutar comandos de la CLI y una API para control remoto.
La descripción completa y ejemplos de uso se encuentran en
[docs/extra_features.md](docs/extra_features.md).

## Funcionalidades extra

TradeBot incluye una serie de capacidades adicionales más allá del MVP
original. Entre ellas se destacan las estrategias de arbitraje
triangular y entre exchanges, señales basadas en microestructura,
adaptadores para múltiples venues con soporte de testnet, un panel web
que permite ejecutar comandos de la CLI y una API para control remoto.
La descripción completa y ejemplos de uso se encuentran en
[docs/extra_features.md](docs/extra_features.md).

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
# Opcional: habilita el backtester vectorizado
pip install "vectorbt>=0.26"
cp .env.example .env   # completa con tus claves
```
## Arranque rápido

Inicia y detén los servicios de Docker con el Makefile:

```bash
make up    # levanta los servicios en segundo plano
make logs  # sigue los logs de todos los contenedores
make down  # detiene y elimina los servicios
```
## Configuración inicial

1. Copia `.env.example` a `.env` y completa tus claves API (`BINANCE_KEY`,
   `BINANCE_SECRET`, etc.). Para pruebas en modo papel puedes dejar los
   valores vacíos.
2. (Opcional) Levanta la base de datos y el stack de monitoreo con Docker:

   ```bash
   make up
   ```

3. Inicia la API y los dashboards:

   ```bash
   uvicorn tradingbot.apps.api.main:app --reload --port 8000
   ```

   `http://localhost:8000/` abre el panel de credenciales,
   `http://localhost:8000/monitor` el de monitoreo y
   `http://localhost:8000/bots` la gestión de bots.

## Comandos CLI

Todos los comandos están disponibles tanto desde la terminal como desde la
consola del panel web.

```bash
python -m tradingbot.cli <comando> [opciones]
```

| Comando | Descripción | Ejemplo |
|---------|-------------|---------|
| `ingest` | Stream de order book a la base de datos | `python -m tradingbot.cli ingest --venue binance_spot --symbol BTC/USDT --depth 20` |
| `ingest-historical` | Descarga histórica desde Kaiko o CoinAPI | `python -m tradingbot.cli ingest-historical kaiko BTC/USDT --kind trades` |
| `run-bot` | Ejecuta el bot en vivo o testnet | `python -m tradingbot.cli run-bot --exchange binance --symbol BTC/USDT` |
| `paper-run` | Ejecuta una estrategia en modo simulación | `python -m tradingbot.cli paper-run --symbol BTC/USDT --strategy breakout_atr --config params.yaml` |
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
| `cfg-validate` | Valida archivos YAML y reporta campos faltantes | `python -m tradingbot.cli cfg-validate data/examples/backtest.yaml` |

La salida de cada comando aparecerá tanto en la terminal como en la consola
del panel web.

Las estrategias que lo permitan aceptan parámetros externos a través de un
archivo YAML pasado con el flag `--config`.

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

