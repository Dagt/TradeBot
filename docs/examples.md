# Ejemplos de Uso

## Ingesta de datos
```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
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

## Backtest rápido
```bash
python -m tradingbot.cli backtest data/examples/btcusdt_1m.csv \
    --symbol BTC/USDT --strategy breakout_atr
```
Al finalizar se imprimirá un resumen con estadísticas adicionales como
``sharpe``, ``sortino`` y ``deflated_sharpe_ratio`` junto con ``pnl``,
``fill_rate`` y ``slippage``.

## Estrategia Triple Barrier
Requiere ``scikit-learn`` para entrenar el modelo de gradient boosting.
```bash
python -m tradingbot.cli backtest data/examples/btcusdt_1m.csv \
    --symbol BTC/USDT --strategy triple_barrier
```

## API
```bash
uvicorn tradingbot.apps.api.main:app --reload --port 8000
```

## Notebook
Consulta el notebook [docs/notebooks/breakout_atr.ipynb](notebooks/breakout_atr.ipynb)
para ver un flujo de trabajo completo de un backtest.

## Barrido de parámetros con vectorbt
Se requiere instalar la dependencia opcional `vectorbt`.

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
