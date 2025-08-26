# Gestión de riesgo

## Cálculo de `equity_pct` y `risk_pct`

El capital asignado a cada operación se determina multiplicando la equity
actual por `equity_pct`:

```
notional = equity_total * equity_pct
```

El riesgo máximo permitido sobre esa operación se controla con `risk_pct`:

```
max_loss = notional * risk_pct
```

A partir del notional se calcula la cantidad a comprar o vender en función del
precio del activo.

## Uso de `strength`

Las estrategias pueden emitir señales con un atributo `strength` que escala el
cambio propuesto en la posición. Un valor mayor a `1.0` permite piramidar
agregando tamaño; valores entre `0` y `1` reducen exposición y `0` cierra la
posición.

## DailyGuard y drawdown global

`DailyGuard` monitorea la equity intradía y la racha de pérdidas. Si se
supera la pérdida diaria permitida o el drawdown intradía, el bot se detiene o
cierra posiciones según la configuración. Esto complementa el seguimiento del
drawdown global que realiza el gestor de riesgo.

## Ejemplos

Configuración orientada a una cuenta pequeña operando un activo de bajo valor:

```yaml
backtest:
  data: data/examples/btcusdt_1m.csv   # reemplazar con datos del activo elegido
  symbol: DOGE/USDT
  strategy: breakout_atr
  initial_equity: 100

risk:
  equity_pct: 0.05
  risk_pct: 0.02
```
