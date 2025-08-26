# Gestión de riesgo

## Asignación por señal

Cada señal trae un `strength` que define el tamaño según la fórmula
`notional = equity * strength`. Un valor de `1.0` utiliza todo el capital
disponible, valores mayores piramidan la posición y menores la reducen.
Por ejemplo, `strength = 1.5` incrementa la exposición un 50 %, mientras que
`strength = 0.5` la reduce a la mitad. El campo `risk_pct` actúa como
stop‑loss local: la posición se cierra si la pérdida supera `notional * risk_pct`.

## PortfolioGuard

Para limitar el uso de capital se puede emplear `PortfolioGuard`, que permite fijar límites globales (`total_cap_pct`) o por símbolo (`per_symbol_cap_pct`).

## DailyGuard y drawdown global

`DailyGuard` monitorea la equity intradía y la racha de pérdidas. Si se supera la pérdida diaria permitida o el drawdown intradía, el bot se detiene o cierra posiciones según la configuración. Esto complementa el seguimiento del drawdown global que realiza el gestor de riesgo.

## Ejemplos

Configuración orientada a una cuenta pequeña operando un activo de bajo valor:

```yaml
backtest:
  data: data/examples/btcusdt_1m.csv   # reemplazar con datos del activo elegido
  symbol: DOGE/USDT
  strategy: breakout_atr
  initial_equity: 100
```
