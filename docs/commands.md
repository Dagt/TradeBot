# Comandos de la CLI

Todos los comandos se ejecutan como:

```bash
python -m tradingbot.cli <comando> [opciones]
```

A continuación se describen los comandos disponibles.

## `ingest`
Recibe datos de mercado en vivo y opcionalmente los almacena.
- `--venue`: intercambio a utilizar (ej. `binance_spot`, `binance_futures_ws`, `bybit_futures_ws`, `okx_futures_ws`). Los nombres siguen el patrón `<exchange>_<market>` para REST y `<exchange>_<market>_ws` para WebSocket; se añade `_testnet` automáticamente cuando se usa el entorno de prueba.
- `--symbol`: puede repetirse para varios pares (por defecto `BTC/USDT`).
- `--depth`: profundidad del libro de órdenes (10).
- `--kind`: tipo de dato: `trades`, `trades_multi`, `orderbook`, `bba`, `delta`, `funding`, `oi`.
- `--persist`: si se indica, guarda los datos en la base de datos.
- `--backend`: backend de almacenamiento (`timescale` o `csv`).

Ejemplos:

```bash
# Funding rate en Binance Futures
python -m tradingbot.cli ingest --venue binance_futures_ws --symbol BTC/USDT --kind funding

# Open interest en Bybit
python -m tradingbot.cli ingest --venue bybit_futures_ws --symbol BTC/USDT --kind open_interest

# Funding y open interest en OKX
python -m tradingbot.cli ingest --venue okx_futures_ws --symbol BTC/USDT --kind funding
python -m tradingbot.cli ingest --venue okx_futures_ws --symbol BTC/USDT --kind open_interest

# Trades de múltiples símbolos
python -m tradingbot.cli ingest --venue binance_spot_ws --symbol BTC/USDT --symbol ETH/USDT --kind trades_multi
```

### Uso en mainnet

Por defecto, los conectores de Binance Futures usan el entorno de prueba. Para
conectarse a la red principal se debe deshabilitar el modo testnet mediante la
variable de entorno `BINANCE_FUTURES_TESTNET`:

```bash
BINANCE_FUTURES_TESTNET=false python -m tradingbot.cli ingest --venue binance_futures_ws --symbol BTC/USDT --kind funding
```

## `backfill`
Descarga datos históricos con límites de velocidad.
- `--days`: número de días hacia atrás (1 por defecto).
- `--symbols`: lista de símbolos a descargar.
- `--start`: fecha inicial en formato ISO.
- `--end`: fecha final en formato ISO.

## `ingest-historical`
Obtiene datos históricos de Kaiko o CoinAPI.
- `source`: `kaiko` o `coinapi`.
- `symbol`: par de mercado.
- `--exchange`: requerido para Kaiko.
- `--kind`: `trades`, `orderbook`, `open_interest` o `funding`.
- `--backend`: backend de almacenamiento (por defecto `timescale`).
- `--limit`: cantidad de registros a traer.
- `--depth`: profundidad del order book.

## `run-bot`
Ejecuta el bot en modo en vivo (testnet o real).
- `--exchange`: nombre del exchange (`binance`).
- `--market`: `spot` o `futures`.
- `--symbol`: puede repetirse; símbolo a operar.
- `--testnet`: usa endpoints de prueba.
- `--trade-qty`: tamaño de la orden.
- `--leverage`: apalancamiento para futuros.
- `--dry-run`: simula órdenes en testnet.
- `--stop-loss` y `--take-profit`: porcentajes de la estrategia.
- `--stop-loss-pct` y `--max-drawdown-pct`: límites del gestor de riesgo.

## `paper-run`
Corre una estrategia en modo paper (sin dinero real) y expone métricas.
- `--symbol`: par a operar (por defecto `BTC/USDT`).
- `--strategy`: nombre de la estrategia (`breakout_atr`).
- `--metrics-port`: puerto para las métricas (8000).
- `--config`: ruta a un archivo YAML con parámetros opcionales.

## `real-run`
Ejecuta el bot contra un exchange real.
- `--exchange`: nombre del exchange.
- `--market`: `spot` o `futures`.
- `--symbol`: puede repetirse.
- `--trade-qty`: tamaño de la orden.
- `--leverage`: apalancamiento.
- `--dry-run`: simula órdenes sin enviarlas.
- `--i-know-what-im-doing`: confirmación necesaria para operar con dinero real.

## `daemon`
Lanza el demonio principal usando una configuración de Hydra.
- `config`: ruta al YAML de configuración (por defecto `config/config.yaml`).

## `ingestion-workers`
Inicia trabajadores de ingesta de funding y open interest definidos en un YAML.
- `config`: ruta al archivo de configuración.
- `--backend`: backend de almacenamiento.

## `cfg-validate`
Valida que un archivo YAML de configuración contenga los campos mínimos
necesarios para backtesting y walk-forward.

## `backtest`
Ejecuta un backtest vectorizado desde un archivo CSV.
- `data`: ruta al CSV.
- `--symbol`: par a evaluar.
- `--strategy`: estrategia a utilizar.

## `backtest-cfg`
Ejecuta un backtest basado en un archivo de configuración Hydra.
- `config`: archivo YAML con los parámetros.

## `backtest-db`
Realiza un backtest usando datos almacenados en la base de datos.
- `--exchange`: nombre del exchange.
- `--symbol`: par a evaluar.
- `--strategy`: estrategia.
- `--start` y `--end`: rango de fechas (YYYY-MM-DD).
- `--timeframe`: periodo de las velas (`1m`).

## `walk-forward`
Optimiza una estrategia con técnica walk-forward usando una configuración
Hydra.
- `config`: archivo YAML con datos, estrategia y grilla de parámetros.

## `report`
Muestra un resumen de PnL desde TimescaleDB.
- `--venue`: nombre del venue (por defecto `binance_spot_testnet`).

## `train-ml`
Entrena un modelo basado en `MLStrategy` y lo guarda en disco.
- `data`: CSV con los datos de entrenamiento.
- `target`: columna objetivo.
- `output`: archivo donde se almacenará el modelo.

## `tri-arb`
Ejecuta un arbitraje triangular simple en Binance.
- `route`: cadena `BASE-MID-QUOTE` que define los tres pares.
- `--notional`: monto a utilizar en la divisa de salida.

## `cross-arb`
Arbitraje entre un mercado spot y uno de futuros.
- `symbol`: par a arbitrar.
- `spot`: nombre del adaptador spot.
- `perp`: nombre del adaptador de futuros.
- `--threshold`: diferencia mínima de precio para actuar.
- `--notional`: monto por pata.

## `run-cross-arb`
Versión que utiliza el `ExecutionRouter` para arbitraje spot/perp.
Acepta los mismos parámetros que `cross-arb`.

