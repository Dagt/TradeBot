# Estructura del CSV de fills

El motor de backtesting puede exportar cada fill ejecutado en un archivo CSV.
Cada fila representa una operación ejecutada e incluye la siguiente información:

| Columna | Descripción |
| --- | --- |
| `timestamp` | Marca de tiempo del bar donde se ejecutó el fill |
| `reason` | Motivo del fill. Valores posibles: `order` (ejecución de la orden), `take_profit`, `stop_loss`, `trailing_stop` o `exit` (cierre manual u otros motivos) |
| `side` | `buy` o `sell` |
| `price` | Precio de ejecución |
| `qty` | Cantidad ejecutada |
| `strategy` | Estrategia que generó la orden |
| `symbol` | Símbolo negociado |
| `exchange` | Venue en el que se ejecutó |
| `fee_cost` | Coste de comisión pagado |
| `slippage_pnl` | PnL debido al slippage (positivo si es favorable) |
| `realized_pnl` | PnL realizado del fill, neto de comisiones y slippage |
| `realized_pnl_total` | PnL realizado acumulado hasta este fill |
| `equity_after` | Equity total tras el fill |

Estas columnas permiten auditar el resultado de la simulación y analizar el
impacto de costes y deslizamientos en cada operación.

El campo `realized_pnl_total` coincide con el valor acumulado expuesto por
`RiskManager.pos.realized_pnl`, que incluye las comisiones y el efecto del
slippage.
