# Momentum 5 m reference backtest

Este experimento compara el encolado limit de la estrategia Momentum antes y después del ajuste para permanecer a ≤1 tick/≤0.1 ATR del mejor bid/ask. Para garantizar la reproducibilidad se utiliza el script [`docs/backtests/momentum_5m_reference.py`](backtests/momentum_5m_reference.py), que genera una ventana sintética de 5 min y ejecuta un backtest vectorial del motor discreto.

## Cómo reproducir

```bash
python docs/backtests/momentum_5m_reference.py
```

## Resultados

| Configuración | Órdenes | Fills | Fee medio por fill (USD) |
| --- | --- | --- | --- |
| Legacy (`_legacy_config`) | 14 | 0 | 0.0000 |
| Actualizada (`_configure_limit`) | 14 | 4 | 0.0999 |

La revisión mantiene la orden inicial pegada al mejor precio y, aun así, permite perseguir el movimiento. En el escenario de 5 m los fills pasan de 0 a 4 mientras que el fee medio por fill cae a 0.10 USD gracias al mayor porcentaje de ejecuciones maker.【060ba2†L1-L3】
