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

## API
```bash
uvicorn tradingbot.apps.api.main:app --reload --port 8000
```

## Notebook
Consulta el notebook [docs/notebooks/breakout_atr.ipynb](notebooks/breakout_atr.ipynb)
para ver un flujo de trabajo completo de un backtest.
