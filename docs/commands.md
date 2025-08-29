# Comandos de la CLI

Todos los comandos se ejecutan como:

```bash
python -m tradingbot.cli <comando> [opciones]
```

A través del endpoint de la API `/cli/start` es posible lanzar estos
comandos en segundo plano. La petición acepta un parámetro opcional
`timeout` (en segundos) que limita la duración máxima del proceso. Para
backtests muy largos se recomienda establecer un `timeout` alto o `null`
para deshabilitar el límite.

A continuación se describen los comandos disponibles. Todas las estrategias
emiten señales con un campo `strength`. El `RiskService` utiliza esa señal para
dimensionar automáticamente la posición (`notional = equity * strength`).
Valores mayores a `1.0` piramidan la exposición, menores la desescalan. El
parámetro `risk_pct` establece la pérdida máxima permitida y `vol_target`
dimensiona la posición según la volatilidad.

Ejemplo de configuración de riesgo:

```yaml
risk:
  risk_pct: 0.02
  vol_target: 0.01
  total_cap_pct: null
  per_symbol_cap_pct: null
```

## `exchange_configs`

Las configuraciones específicas de cada venue se declaran en `config/config.yaml`
 bajo la clave `exchange_configs`. Cada entrada define el tipo de mercado,
 las comisiones maker/taker y el tamaño mínimo de tick a aplicar durante los backtests y al
 utilizar la CLI. Se pueden definir entradas adicionales con el sufijo `_testnet`
 para ajustar tarifas en entornos de prueba:

```yaml
exchange_configs:
  binance_spot:
    market_type: spot
    maker_fee_bps: 10.0
    taker_fee_bps: 10.0
    tick_size: 0.01
  okx_spot:
    market_type: spot
    maker_fee_bps: 8.0
    taker_fee_bps: 10.0
    tick_size: 0.1
  okx_spot_testnet:
    market_type: spot
    maker_fee_bps: 8.0
    taker_fee_bps: 10.0
    tick_size: 0.1
```

Si un venue no aparece explícitamente, el tipo de mercado se infiere
automáticamente: los nombres que terminan en `_spot` se consideran spot y el
resto se tratan como futuros/perpetuos.

## `ingest`
Recibe datos de mercado en vivo y opcionalmente los almacena.
- `--venue`: intercambio a utilizar (ej. `binance_spot`, `binance_futures_ws`, `bybit_futures_ws`, `okx_futures_ws`). Los nombres siguen el patrón `<exchange>_<market>` para REST y `<exchange>_<market>_ws` para WebSocket; se añade `_testnet` automáticamente cuando se usa el entorno de prueba.
- `--symbol`: puede repetirse para varios pares (por defecto `BTC/USDT`).
- `--depth`: profundidad del libro de órdenes (10).
- `--kind`: tipo de dato: `trades`, `trades_multi`, `orderbook`, `bba`, `delta`, `funding`, `oi`.
- `--persist`: si se indica, guarda los datos en la base de datos.
- `--backend`: backend de almacenamiento (`timescale` o `csv`).

Nota: Bybit no provee streams de `funding` ni `open_interest` vía WebSocket;
para esos datos utilice el adaptador REST (`bybit_futures`).

Ejemplos:

```bash
# Funding rate en Binance Futures
python -m tradingbot.cli ingest --venue binance_futures_ws --symbol BTC/USDT --kind funding

# Open interest en Bybit (usar REST)
python -m tradingbot.cli ingest --venue bybit_futures --symbol BTC/USDT --kind open_interest

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
- `--venue`: nombre del venue (`binance_spot`, `binance_futures`, etc.).
- `--start`: fecha inicial en formato ISO.
- `--end`: fecha final en formato ISO.

## `ingest-historical`
Obtiene datos históricos de Kaiko o CoinAPI.
- `source`: `kaiko` o `coinapi`.
- `symbol`: par de mercado.
- `--exchange`: requerido para Kaiko.
- `--kind`: `trades`, `orderbook`, `bba`, `book_delta`, `open_interest` o `funding`.
- `--backend`: backend de almacenamiento (por defecto `timescale`).
- `--limit`: cantidad de registros a traer.
- `--depth`: profundidad del order book.

Para establecer las credenciales, use:

```
secrets set kaiko <key> <secret>
secrets set coinapi <key> <secret>
```

Como alternativa, las variables de entorno `KAIKO_API_KEY` y `COINAPI_KEY`
pueden definirse con los valores correspondientes.

## `run-bot`
Ejecuta el bot en modo en vivo (testnet o real).
- `--venue`: nombre del venue (ej. `binance_spot`, `okx_futures`).
- `--symbol`: puede repetirse; símbolo a operar.
- `--testnet`: usa endpoints de prueba.
- `--risk-pct`: porcentaje de pérdida máxima del equity asignado.
- `--leverage`: apalancamiento para futuros.
- `--dry-run`: simula órdenes en testnet.
- `--take-profit`: porcentaje de toma de ganancias.

## `paper-run`
Corre una estrategia en modo paper (sin dinero real) y expone métricas.
- `--symbol`: par a operar (por defecto `BTC/USDT`).
- `--strategy`: nombre de la estrategia (`breakout_atr`).
- `--metrics-port`: puerto para las métricas (8000).
- `--config`: ruta a un archivo YAML con parámetros opcionales.

## `real-run`
Ejecuta el bot contra un exchange real.
- `--venue`: nombre del venue.
- `--symbol`: puede repetirse.
- `--risk-pct`: porcentaje de pérdida máxima del equity asignado.
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

Los comandos de backtesting solo aceptan venues en modo live con sufijo
`*_spot` o `*_futures`; los venues que terminan en `_testnet` o `_ws` son
rechazados.

## `backtest`
Ejecuta un backtest vectorizado desde un archivo CSV.
- `data`: ruta al CSV.
- `--symbol`: par a evaluar.
- `--strategy`: estrategia a utilizar.
- `--fills-csv PATH`: exporta los fills a un CSV.

Si se especifica `--fills-csv`, se genera un archivo con las columnas
`timestamp, side, price, qty, strategy, symbol, exchange, fee, cash_after, base_after, equity_after, realized_pnl`.
Desde este archivo puede reconstruirse el efectivo y la posición para validar el PnL final:

```python
import pandas as pd

fills = pd.read_csv("fills.csv")
# `base_after`, `cash_after` y `equity_after` ya contienen los balances tras cada fill.
```

La última fila de `cash` y `position` debe coincidir con los valores reportados
por el motor, permitiendo verificar el PnL obtenido.

## `backtest-cfg`
Ejecuta un backtest basado en un archivo de configuración Hydra.
- `config`: archivo YAML con los parámetros.
- `--fills-csv PATH`: exporta los fills a un CSV.

## `backtest-db`
Realiza un backtest usando datos almacenados en la base de datos.
- `--venue`: nombre del venue.
- `--symbol`: par a evaluar.
- `--strategy`: estrategia.
- `--start` y `--end`: rango de fechas (YYYY-MM-DD).
- `--timeframe`: periodo de las velas (`1m`).
- `--fills-csv PATH`: exporta los fills a un CSV.

## `walk-forward`
Optimiza una estrategia con técnica walk-forward usando una configuración
Hydra.
- `config`: archivo YAML con datos, estrategia y grilla de parámetros.
- `--fills-csv PATH`: exporta los fills del periodo de prueba a un CSV.

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

El tamaño de la operación se calcula automáticamente a partir de la señal
`strength` y el `RiskService`.

## `cross-arb`
Arbitraje entre un mercado spot y uno de futuros.
- `symbol`: par a arbitrar.
- `spot`: nombre del adaptador spot.
- `perp`: nombre del adaptador de futuros.
- `--threshold`: diferencia mínima de precio para actuar.

El tamaño de cada pata se calcula de forma automática a partir del diferencial
(`edge`) y el capital disponible reportado por `broker.equity()`.

## `run-cross-arb`
Versión que utiliza el `ExecutionRouter` para arbitraje spot/perp.
Acepta los mismos parámetros que `cross-arb` y dimensiona las órdenes de la
misma manera.

