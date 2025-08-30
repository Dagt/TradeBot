# Gestión de riesgo

## RiskManager universal

El módulo [`core/risk_manager.py`](../src/tradingbot/core/risk_manager.py)
implementa un gestor de riesgo genérico que opera junto a
[`core/account.py`](../src/tradingbot/core/account.py). Su interfaz expone:

- `calc_position_size(signal_strength, price)`: dimensiona la posición según la
  fuerza de la señal y el riesgo por operación.
- `initial_stop(entry_price, side, atr)`: calcula el stop inicial usando un
  múltiplo de ATR.
- `update_trailing(trade, current_price, fees_slip=0.0)`: aplica un trailing
  stop adaptativo en tres etapas.
- `manage_position(trade, signal=None)`: decide si mantener o cerrar una
  posición según el stop y la señal actual.
- `check_global_exposure(symbol, new_alloc)`: verifica que la nueva exposición
  no supere el límite global para el símbolo.

### Ejemplo de uso

```python
from tradingbot.core.account import Account
from tradingbot.core.risk_manager import RiskManager

account = Account(max_symbol_exposure=1000.0, cash=1000.0)
rm = RiskManager(account, risk_per_trade=0.02)  # equivalente a --risk-pct 2

signal = {"side": "buy", "strength": 0.6, "atr": 5}
price = 100
size = rm.calc_position_size(signal["strength"], price)
if rm.check_global_exposure("BTC/USDT", size * price):
    stop = rm.initial_stop(price, signal["side"], atr=signal["atr"])
    trade = {
        "side": "buy",
        "entry_price": price,
        "qty": size,
        "stop": stop,
        "atr": signal["atr"],
    }
    rm.update_trailing(trade, current_price=112)
    trade["current_price"] = 112
    decision = rm.manage_position(trade)
```

`signal["strength"]` acepta valores continuos y escala el tamaño de la orden.
El método `update_trailing` mueve el stop a *break-even*, asegura 1 USD neto y
luego sigue al precio a `2 × ATR`. `manage_position` decide si mantener o cerrar
la operación y `check_global_exposure` valida el límite global por símbolo.

## Asignación por señal

Cada señal trae un `strength` que define el tamaño según la fórmula
`notional = equity * strength`. Un valor de `1.0` utiliza todo el capital
disponible, valores mayores piramidan la posición y menores la reducen.
Por ejemplo, `strength = 1.5` incrementa la exposición un 50 %, mientras que
`strength = 0.5` la reduce a la mitad. El campo `risk_pct` establece la pérdida
máxima permitida y `vol_target` dimensiona la posición según la volatilidad.

El parámetro `risk_pct` debe estar entre 0 y 1. Los valores de 1 a 100 se
interpretan como porcentajes y se convierten dividiéndolos entre 100 (por
ejemplo, un valor de 1 se normaliza a 0.01). Valores negativos o mayores a 100
provocan un error.

Ejemplo desde el CLI:

```bash
python -m tradingbot.cli backtest data.csv --risk-pct 2   # 2 % de riesgo
# también puede usarse "--risk-pct 1" para un 1 % de riesgo
```

## PortfolioGuard

Para limitar el uso de capital se puede emplear `PortfolioGuard`, que permite fijar límites globales (`total_cap_pct`) o por símbolo (`per_symbol_cap_pct`). Estos valores pueden establecerse en `null` para deshabilitar los límites.

## DailyGuard y drawdown global

`DailyGuard` monitorea la equity intradía y la racha de pérdidas. Si se supera la pérdida diaria permitida o el drawdown intradía, el bot se detiene o cierra posiciones según la configuración. Esto complementa el seguimiento del drawdown global que realiza el gestor de riesgo.

## Ejemplos

Configuración orientada a una cuenta pequeña operando un activo de bajo valor:

```yaml
backtest:
  data: data/examples/btcusdt_3m.csv   # reemplazar con datos del activo elegido
  symbol: DOGE/USDT
  strategy: breakout_atr
  initial_equity: 100

risk:
  risk_pct: 0.02
  vol_target: 0.01
  total_cap_pct: null
  per_symbol_cap_pct: null
```
