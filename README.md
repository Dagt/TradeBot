# TradeBot

TradeBot es un bot de trading educativo y de código abierto para mercados de
criptomonedas. El proyecto incluye herramientas para descargar datos,
realizar backtesting, ejecutar estrategias en modo prueba o con dinero real
y monitorear el desempeño con tableros de control.

La documentación completa está en la carpeta [`docs`](docs/). Allí se
explican los conceptos básicos, las estrategias disponibles y todas las
opciones de la línea de comandos.

## Requisitos mínimos

- Python 3.11
- Dependencias listadas en `requirements.txt`

## Instalación rápida

```bash
pip install -e .
```

## Uso básico

Todos los comandos se ejecutan con `python -m tradingbot.cli` seguido del
subcomando deseado. Por ejemplo, para ver la ayuda general:

```bash
python -m tradingbot.cli --help
```

Para una guía detallada de cada comando, consulte [docs/commands.md](docs/commands.md).

## Recursos

- [Blueprint de arquitectura](BLUEPRINT.md)
- [MVP: características mínimas](MVP.md)
- [Glosario de términos](docs/glossary.md)
- [Estrategias disponibles](docs/strategies.md)
- [Dashboards de monitoreo](docs/dashboards.md)

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
