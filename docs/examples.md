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
