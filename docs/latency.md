# Impacto de los acumuladores incrementales

El reemplazo de `df.rolling(...).quantile()` por acumuladores basados en
`deque` y `bisect` reduce de forma significativa la latencia de cálculo.

En un benchmark sobre 10 000 muestras con ventana 100:

```
>>> pandas 25.07s
>>> incremental 0.01s
```

Esto supone una reducción de **más de tres órdenes de magnitud** en el
cálculo de percentiles, mejorando la capacidad de respuesta de las
estrategias en tiempo real.
