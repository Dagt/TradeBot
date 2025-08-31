# Estrategias Disponibles

Las estrategias definen cómo se toman decisiones de compra o venta. A
continuación se presenta un resumen en lenguaje sencillo.

Todas las señales incluyen un campo `strength` continuo que dimensiona las
órdenes mediante `notional = equity * strength`. Opcionalmente pueden definir
`limit_price` para cotizar órdenes *limit* a través del broker. El
`RiskService` combina `RiskManager` y `CoreRiskManager` para calcular el tamaño
y aplicar un trailing stop adaptativo.

Ejemplo de señal:

```python
signal = {"side": "buy", "strength": 0.75, "limit_price": 100.5}
```

Las estrategias pueden implementar callbacks `on_partial_fill` y
`on_order_expiry` para decidir si re‑cotizar, cancelar o convertir el remanente
en una orden de mercado cuando la ventaja desaparece.

### Breakout con ATR (`breakout_atr`)
Compra cuando el precio supera el canal superior calculado con el indicador
ATR y vende cuando cae por debajo del canal inferior.

### Breakout por Volumen (`breakout_vol`)
Detecta rupturas de precio acompañadas de incrementos de volumen.

### Momentum
Sigue la tendencia actual: compra si el precio sube de forma sostenida y
vende cuando la tendencia se revierte.

### Mean Reversion (`mean_reversion`)
Asume que los precios tienden a volver a un promedio. Compra cuando el
precio cae por debajo del promedio y vende cuando lo supera.

### Mean Reversion con OFI (`mean_rev_ofi`)
Variación de mean reversion que utiliza el desequilibrio de flujo de órdenes
(Order Flow Imbalance) para estimar el regreso al promedio.

### Depth Imbalance (`depth_imbalance`)
Evalúa el desequilibrio entre órdenes de compra y venta en el libro de
órdenes para anticipar movimientos.

### Liquidity Events (`liquidity_events`)
Busca momentos donde la liquidez cambia rápidamente, lo que puede generar
oportunidades de corto plazo.

### Order Flow (`order_flow`)
Analiza el flujo de órdenes que llegan al mercado para detectar presiones de
compra o venta.

### Triple Barrier (`triple_barrier`)
Emplea tres barreras (objetivo, límite de pérdida y tiempo) únicamente para
etiquetar los datos; la gestión de la posición (stops y cierres) la realiza el
`RiskService`.

### Cash and Carry (`cash_and_carry`)
Aprovecha diferencias de precio entre el mercado spot y los futuros para
capturar rendimientos sin exposición direccional.

### Arbitraje (`arbitrage`)
Busca beneficios cuando un mismo activo tiene precios distintos en dos
mercados.

### Arbitraje Triangular (`arbitrage_triangular`)
Opera tres pares de divisas al mismo tiempo para explotar desequilibrios en
las tasas de cambio.

### Arbitraje entre Exchanges (`cross_exchange_arbitrage`)
Compara el precio entre un mercado spot y uno de futuros (perpetuo) y opera
cuando la diferencia supera un umbral.
