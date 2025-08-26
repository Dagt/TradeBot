# Gestión de riesgo

## Cálculo del notional y `risk_pct`

La exposición de cada operación se determina a partir de la equity actual:

```
notional = equity * strength
```

`strength` es la fracción del capital que se desea asignar. Valores mayores a
`1` piramidan la posición; valores entre `0` y `1` desescalan y `0` cierra la
exposición. El stop‑loss local se define como porcentaje de ese notional usando
`risk_pct`:

```
max_loss = notional * risk_pct
```

A partir del notional se calcula la cantidad a comprar o vender en función del
precio del activo.

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
    risk_pct: 0.02
```

Las señales deben indicar `strength` para asignar capital; por ejemplo,
`strength = 0.05` utiliza el 5 % del equity disponible.
