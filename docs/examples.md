# Ejemplos de Uso

## Ingesta de datos
```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
```

### Ingesta histórica
```bash
# Trades desde Kaiko
python -m tradingbot.cli ingest-historical kaiko BTC-USDT --exchange binance --kind trades
# Order books desde CoinAPI
python -m tradingbot.cli ingest-historical coinapi BTC/USD --kind orderbook --depth 20
```

## Trading en vivo en Testnet
```bash
python -m tradingbot.cli run-bot --exchange binance --market spot --symbol BTC/USDT
python -m tradingbot.cli run-bot --exchange binance --market futures --symbol BTC/USDT --leverage 5
```

## Backfill y backtesting vectorizado
1. Levantar una base de datos Timescale o Quest:
```bash
./bin/start_timescale.sh    # o ./bin/start_questdb.sh
```
2. Descargar datos de mercado usando la utilidad CLI:
```bash
# OHLCV
./bin/market_data.py ohlcv binance BTC/USDT 2024-01-01T00:00:00 2024-01-02T00:00:00
# Trades
./bin/market_data.py trades binance BTC/USDT 2024-01-01T00:00:00 2024-01-01T01:00:00
# Orderbook L2
./bin/market_data.py l2 binance BTC/USDT --snapshots 60 --interval 1.0
```
3. Cargar las barras en pandas y ejecutar un backtest vectorizado:
```python
import pandas as pd
from tradingbot.storage import timescale
from tradingbot.backtesting.vectorbt_wrapper import run_parameter_sweep

engine = timescale.get_engine()
bars = pd.read_sql_query(
    "SELECT ts, o, h, l, c, v FROM market.bars WHERE symbol='BTC/USDT' ORDER BY ts",
    engine, parse_dates=["ts"],
).set_index("ts")

def ma_signal(close, fast, slow):
    import vectorbt as vbt
    fast_ma = vbt.MA.run(close, fast)
    slow_ma = vbt.MA.run(close, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return entries, exits

stats = run_parameter_sweep(bars, ma_signal, {"fast": [5], "slow": [20]})
print(stats)
```

Más ejemplos de estrategias avanzadas y uso del panel web se encuentran
en [extra_features.md](extra_features.md).

## Backtest rápido
```bash
python -m tradingbot.cli backtest data/examples/btcusdt_1m.csv \
    --symbol BTC/USDT --strategy breakout_atr
```
Al finalizar se imprimirá un resumen con estadísticas adicionales como
``sharpe``, ``sortino`` y ``deflated_sharpe_ratio`` junto con ``pnl``,
``fill_rate`` y ``slippage``.  También se calculan pruebas de estrés básicas
que estiman el capital final ante caídas uniformes del 5% y 10% en cada
período.

## Estrategia Triple Barrier
Requiere ``scikit-learn`` para entrenar el modelo de gradient boosting.
```bash
python -m tradingbot.cli backtest data/examples/btcusdt_1m.csv \
    --symbol BTC/USDT --strategy triple_barrier
```

## Entrenar modelo ML
```bash
python -m tradingbot.cli train-ml datos.csv objetivo modelo.joblib
```

## API
```bash
uvicorn tradingbot.apps.api.main:app --reload --port 8000
```

Un dashboard estático en `monitoring/static/index.html` consulta estos
endpoints para mostrar métricas y estado de las estrategias.

## Notebook
Consulta el notebook [docs/notebooks/breakout_atr.ipynb](notebooks/breakout_atr.ipynb)
para ver un flujo de trabajo completo de un backtest.

## Barrido de parámetros con vectorbt
Se requiere instalar la dependencia opcional `vectorbt` (por ejemplo, `pip install "vectorbt>=0.26"`).

```python
import numpy as np
import pandas as pd
from tradingbot.backtesting.vectorbt_wrapper import run_parameter_sweep

def ma_signal(close, fast, slow):
    import vectorbt as vbt
    fast_ma = vbt.MA.run(close, fast)
    slow_ma = vbt.MA.run(close, slow)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    return entries, exits

price = pd.Series(np.linspace(1, 2, 100))
data = pd.DataFrame({"close": price})
params = {"fast": [5, 10], "slow": [20]}

stats = run_parameter_sweep(data, ma_signal, params)
print(stats)
```

## Estrategias Freqtrade

Requiere instalar la dependencia opcional `freqtrade`.

```python
import pandas as pd
from tradingbot.backtesting.freqtrade_wrapper import load_ohlcv, run_strategy


class SimpleStrategy:
    timeframe = "1m"

    def populate_indicators(self, dataframe, metadata):
        return dataframe

    def populate_entry_trend(self, dataframe, metadata):
        dataframe["enter_long"] = dataframe["close"] > dataframe["close"].shift(1)
        return dataframe

    def populate_exit_trend(self, dataframe, metadata):
        dataframe["exit_long"] = dataframe["close"] < dataframe["close"].shift(1)
        return dataframe


data = load_ohlcv("data/examples/btcusdt_1m.csv")
result = run_strategy(data, SimpleStrategy)
print(result["equity"])
```

## Walk-forward

```python
from tradingbot.backtesting.engine import walk_forward_optimize

grid = [{"rsi_n": 10, "rsi_threshold": 55}, {"rsi_n": 14, "rsi_threshold": 60}]
wf_results = walk_forward_optimize(
    "data/examples/btcusdt_1m.csv",
    "BTC/USDT",
    "momentum",
    grid,
    train_size=1000,
    test_size=250,
)
print(wf_results)
```

## Optimización walk-forward con MLflow

La utilidad `bin/walk_forward_example.py` ejecuta un ciclo de optimización
por ventana y registra los resultados en MLflow.  Primero inicia la interfaz
de usuario de MLflow:

```bash
mlflow ui --backend-store-uri ./mlruns
```

En otra terminal ejecuta la optimización:

```bash
python bin/walk_forward_example.py data/examples/btcusdt_1m.csv BTC/USDT \
    --mlflow-uri ./mlruns --experiment walk-forward-demo
```

Cada ventana queda registrada como un run separado con métricas como
`test_equity`, `test_sharpe` y `test_drawdown`.  En el dashboard se pueden
comparar fácilmente los resultados promedio y por ventana.
