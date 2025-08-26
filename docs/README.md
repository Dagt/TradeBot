# Documentación de TradeBot

Esta carpeta contiene material de referencia para entender y usar el bot.
Si es la primera vez que lo utiliza, comience por revisar el
[glosario](glossary.md) y luego la [lista de comandos](commands.md).

## Contenido

- [Glosario](glossary.md)
- [Comandos de la CLI](commands.md)
- [Estrategias disponibles](strategies.md)
- [Dashboards de monitoreo](dashboards.md)

## Ejemplo de parámetros porcentuales

El gestor de riesgo utiliza porcentajes tanto para el cálculo de stop loss como
para el seguimiento (*trailing stop*). Un ejemplo rápido desde la línea de
comandos:

```bash
python -m tradingbot.cli run --stop-loss-pct 0.01 --max-drawdown-pct 0.04
```

Esto aplicará un stop loss del 1 % y reducirá la posición si se produce un
drawdown del 4 % o más.
