# BreakoutVol - Ajuste de offsets por volatilidad

## Resumen del cambio

La estrategia `BreakoutVol` ahora calcula el precio límite desplazándolo en
puntos básicos respecto al último cierre.  El desplazamiento se obtiene a partir
de la volatilidad en bps (`vol_bps`) y del ATR normalizado sobre el precio.  Un
nuevo parámetro `max_offset_pct` (por defecto 1.5 %) evita que el offset crezca
de forma descontrolada durante picos de volatilidad.  El valor final queda
expuesto en `bar["limit_offset"]` y en `bar["context"]` para que el gestor de
riesgo pueda reajustar el tamaño o aplicar límites adicionales.

## Señales para el gestor de riesgo

Las siguientes claves se incluyen en cada barra procesada:

- `limit_offset`: desplazamiento absoluto aplicado al límite.
- `limit_offset_bps`: desplazamiento en puntos básicos.
- `limit_offset_pct`: desplazamiento expresado como porcentaje decimal.

## Parámetros relevantes

| Parámetro          | Descripción                                                        |
|--------------------|--------------------------------------------------------------------|
| `max_offset_pct`   | Tope máximo (fracción del precio) para el offset del límite.       |
| `volatility_factor`| Escala de tamaño en función de la volatilidad medida en bps.       |

Ajustar `max_offset_pct` permite suavizar el impacto de velas extremas sin
renunciar a la sensibilidad que aporta `vol_bps`.

