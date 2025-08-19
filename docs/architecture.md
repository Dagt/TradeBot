# Arquitectura

El proyecto sigue una arquitectura modular orientada a eventos.
Cada componente expone interfaces sencillas para facilitar el
reemplazo o la extensión.

```
src/tradingbot/
├── adapters/      # Acceso a exchanges y fuentes de datos
├── data/          # Ingesta y transformaciones
├── strategies/    # Lógica de trading
├── execution/     # Envío de órdenes y simulaciones
├── risk/          # Gestión de riesgo y salvaguardas
├── storage/       # Persistencia (TimescaleDB, QuestDB, etc.)
├── backtest/      # Motor de backtesting
├── apps/          # Puntos de entrada (API, panel)
└── cli/           # Comandos de línea de uso frecuente
```

Los módulos se comunican mediante un bus de eventos (`bus.py`) lo
que permite desacoplar la generación de señales de la ejecución y
el almacenamiento.

La carpeta `sql/` contiene los esquemas de base de datos y ejemplos
de `docker-compose` para levantar servicios individuales. El stack
completo de demostración se levanta con `docker-compose.yml` en la
raíz del repositorio.

Para una explicación detallada de los componentes y su interacción
consulta [docs/usage.md](usage.md) y los ejemplos de
[docs/examples.md](examples.md).  Las funcionalidades adicionales que
no forman parte del MVP se describen en
[extra_features.md](extra_features.md).
