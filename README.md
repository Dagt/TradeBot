# TradeBot

TradeBot es un bot de trading modular orientado a estrategias intradía,
scalping y arbitraje sobre criptomonedas.  Incluye todo lo necesario para
ingerir datos, realizar backtesting y ejecutar estrategias en modo
``paper`` o real desde una interfaz web.

Un panorama general de la arquitectura se explica en
[blueprint_trading_bot.md](blueprint_trading_bot.md) y la correspondencia con
el código se detalla en
[docs/blueprint_map.md](docs/blueprint_map.md).  El diseño técnico completo
del MVP está en [tradebot_mvp.md](tradebot_mvp.md).

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
   Este comando ejecuta `docker-compose up` y levanta **Prometheus**,
   **Alertmanager** y **Grafana** con dashboards y *data sources*
   preconfigurados, sin pasos manuales posteriores.

4. **Ingerir datos**
   ```bash
   tradingbot ingest
   ```

   (Opcional) para completar históricos:
   ```bash
   tradingbot backfill --days 7 --symbols BTC/USDT
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

## Exchanges y pares soportados

- **Exchanges**: [Binance](https://www.binance.com),
  [Bybit](https://www.bybit.com) y [OKX](https://www.okx.com).  El diseño es
  modular y pueden añadirse más.
- **Mercados**: pares spot y contratos perpetuos disponibles en esos
  exchanges.
- **Pares populares**: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT y cualquier
  otro listado por los exchanges anteriores.

## Estrategias incluidas

Cada estrategia puede correrse en modo **paper trading** o con órdenes
reales.  A continuación se resume la idea y se indica cómo ejecutarla desde
la CLI.  Todas aceptan parámetros opcionales mediante ``--config``.

### Momentum intradía (`momentum`)
**Idea**: seguir la tendencia reciente.  Cuando el RSI supera un umbral y el
OFI confirma presión compradora, se compra; lo inverso genera venta.

```
python -m tradingbot.cli paper-run --strategy momentum --symbol BTC/USDT
```

### Reversión a la media (`mean_reversion`)
**Idea**: los precios vuelven a su media tras desviarse demasiado.
**Implementación**: RSI con niveles superior/inferior para vender o comprar.

### Breakout ATR (`breakout_atr`)
**Idea**: rupturas de un canal de Keltner anuncian movimientos fuertes.
**Implementación**: compra si el cierre supera la banda superior del canal,
vende si cae por debajo de la inferior.

### Breakout de volatilidad (`breakout_vol`)
**Idea**: una subida brusca tras un período tranquilo puede iniciar un
recorrido.  Usa media y desviación estándar.

### Order Flow (`order_flow`)
**Idea**: el promedio del OFI revela desequilibrio de órdenes.
**Implementación**: si el OFI medio es positivo se compra, si es negativo se
vende.

### Mean Reversion OFI (`mean_rev_ofi`)
**Idea**: cuando el z‑score del OFI es extremo y la volatilidad es baja, el
precio suele corregir.

### Depth Imbalance (`depth_imbalance`)
**Idea**: grandes diferencias entre las colas del libro anticipan el
movimiento.

### Eventos de liquidez (`liquidity_events`)
**Idea**: vaciados del libro o gaps amplios indican movimientos inminentes.

### Arbitraje simple (`arbitrage`)
Plantilla para experimentar con spreads entre dos activos.

### Arbitraje triangular (`triangular_arb`)
**Idea**: recorrer rutas A→B→C→A dentro de un exchange para capturar
desalineaciones de precio.

### Arbitraje entre exchanges (`cross_exchange_arbitrage`) y Cash‑and‑Carry (`cash_and_carry`)
**Idea**: aprovechar diferencias entre spot y perp o entre dos exchanges.
El bot abre posiciones opuestas cuando la prima supera un umbral.

### Triple Barrera con ML (`triple_barrier`)
Genera labels de triple barrera y entrena un modelo de **gradient boosting**
para decidir si tomar la señal principal.

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

## Instalación y uso en Windows

1. Instala [Python 3.11](https://www.python.org/downloads/windows/) y [Git](https://git-scm.com/download/win). Durante la instalación de Python marca la casilla **Add Python to PATH**.
2. Abre PowerShell o Git Bash y clona el repositorio:
   ```powershell
   git clone <repo>
   cd TradeBot
   ```
3. Crea y activa un entorno virtual:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
4. Instala las dependencias:
   ```powershell
   pip install -r requirements.txt
   ```
   Opcionalmente habilita el backtester vectorizado:
   ```powershell
   pip install "vectorbt>=0.26"
   ```
5. Copia el archivo de entorno:
   ```powershell
   copy .env.example .env
   ```
6. (Opcional) Si tienes Docker Desktop, inicia los servicios:
   ```powershell
   docker compose up -d
   ```
   También puedes ejecutar `make up` desde Git Bash si dispones de `make`.
7. Ejecuta el bot o la API:
   ```powershell
   python -m tradingbot.cli ingest
   python -m tradingbot.cli backtest
   uvicorn tradingbot.apps.api.main:app --reload --port 8000
   ```
   Luego visita `http://localhost:8000/` para acceder al panel web.

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
| `real-run` | Opera en el exchange real (requiere `--i-know-what-im-doing`) | `python -m tradingbot.cli real-run --exchange binance --symbol BTC/USDT --i-know-what-im-doing` |
| `paper-run` | Ejecuta una estrategia en modo simulación | `python -m tradingbot.cli paper-run --symbol BTC/USDT --strategy breakout_atr --config params.yaml` |
| `daemon` | Levanta el daemon de trading mediante Hydra | `python -m tradingbot.cli daemon config/config.yaml` |
| `backfill` | Backfill de OHLCV y trades con rate limit | `python -m tradingbot.cli backfill --days 7 --symbols BTC/USDT ETH/USDT` |
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

