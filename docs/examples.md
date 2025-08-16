# Ejemplos de Uso

## Ingesta de datos
```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
```

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
