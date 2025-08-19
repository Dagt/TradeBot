# Logging

Los módulos deben obtener un logger mediante `get_logger` para mantener un
formato consistente en toda la aplicación.

```python
from tradingbot.utils.logging import get_logger

log = get_logger(__name__)
```

En las secciones críticas conviene separar el envío de órdenes de la
persistencia de datos y registrar errores específicos.

```python
try:
    res = await adapter.place_order(...)
except Exception:
    log.error("Order placement failed")

try:
    timescale.insert_order(engine, ...)
except Exception:
    log.exception("Persist failure: insert_order")
```

Esto facilita identificar rápidamente el punto exacto del fallo al revisar los
logs.

Para una descripción de funcionalidades avanzadas y cómo generan logs
adicionales (arbitrajes, panel web, API), ver
[extra_features.md](extra_features.md).

