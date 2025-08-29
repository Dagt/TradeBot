# Estructura del CSV de fills

El motor de backtesting puede exportar cada fill ejecutado en un archivo CSV.
Cada fila representa una operación ejecutada e incluye la siguiente información:

| Columna | Descripción |
| --- | --- |
| `timestamp` | Marca de tiempo del bar donde se ejecutó el fill |
| `bar_index` | Índice del bar dentro de la serie de precios |
| `order_id` | Identificador interno de la orden |
| `trade_id` | Secuencia incremental de fills |
| `roundtrip_id` | Identificador del roundtrip al que pertenece el fill |
| `reason` | Motivo del fill (`order`, `take_profit`, `stop_loss`, etc.) |
| `side` | `buy` o `sell` |
| `price` | Precio de ejecución |
| `qty` | Cantidad ejecutada |
| `strategy` | Estrategia que generó la orden |
| `symbol` | Símbolo negociado |
| `exchange` | Venue en el que se ejecutó |
| `fee_type` | Tipo de comisión (`maker` o `taker`) |
| `fee_cost` | Coste de comisión pagado |
| `slip_bps` | Slippage aplicado en puntos básicos |
| `slippage_pnl` | PnL debido al slippage (positivo si es favorable) |
| `cash_after` | Saldo de efectivo tras el fill |
| `base_after` | Posición en unidades del activo tras el fill |
| `equity_after` | Equity total tras el fill |
| `realized_pnl` | PnL realizado del fill, neto de comisiones y slippage |
| `realized_pnl_total` | PnL realizado acumulado hasta este fill |

Estas columnas permiten auditar el resultado de la simulación y analizar el
impacto de costes y deslizamientos en cada operación.

El campo `realized_pnl_total` coincide con el valor acumulado expuesto por
`RiskManager.pos.realized_pnl`, que incluye las comisiones y el efecto del
slippage.
