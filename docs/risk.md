# Gestión de riesgo

## RiskService

[`risk/service.py`](../src/tradingbot/risk/service.py) expone un servicio que
combina `RiskManager`, `CoreRiskManager` y un `Account` para mantener la
exposición bajo control. Además puede integrar un `PortfolioGuard` y un
`DailyGuard` para limitar el uso de capital y las pérdidas intradía.

### Ejemplo de uso

```python
from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService

account = Account(max_symbol_exposure=1000.0, cash=1000.0)
risk = RiskService(
    PortfolioGuard(GuardConfig(venue="demo")),
    account=account,
    risk_pct=0.02,
)

signal = {"side": "buy", "strength": 0.6, "limit_price": 100.0}
price = 100
ok, reason, delta = risk.check_order("BTC/USDT", signal["side"], price, strength=signal["strength"])
if ok and delta > 0:
    stop = risk.initial_stop(price, signal["side"])
    trade = {
        "side": signal["side"],
        "entry_price": price,
        "qty": delta,
        "stop": stop,
        "atr": 5,
    }
    risk.update_trailing(trade, current_price=112)
    decision = risk.manage_position(trade)
```

`signal["strength"]` acepta valores continuos y escala el tamaño de la orden.
El método `update_trailing` mueve el stop a *break-even*, asegura 1 USD neto y
luego sigue al precio a `2 × ATR`. `manage_position` decide si mantener o cerrar
la operación y `check_global_exposure` valida el límite global por símbolo.

Para enviar órdenes *limit* puede utilizarse `broker.place_limit`, que admite
parámetros de *quoting* como `tif` (``GTC``, ``IOC``, ``FOK`` o ``GTD``) y la
bandera `PO` para post-only. Este método también acepta callbacks
`on_partial_fill` y `on_order_expiry` que permiten re‑cotizar el remanente,
cancelar la orden o caer a *market* cuando la ventaja desaparece.

## Asignación por señal

Cada señal trae un `strength` que define el tamaño según la fórmula
`notional = equity * strength`. Un valor de `1.0` utiliza todo el capital
disponible, valores mayores piramidan la posición y menores la reducen.
Por ejemplo, `strength = 1.5` incrementa la exposición un 50 %, mientras que
`strength = 0.5` la reduce a la mitad. El campo `risk_pct` establece la pérdida
máxima permitida.

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
  total_cap_pct: null
  per_symbol_cap_pct: null
```
