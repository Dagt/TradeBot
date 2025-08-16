# Guía de Setup

## Requisitos
- Python 3.11+
- Docker y Docker Compose

## Entorno local
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # completar credenciales
```

Ejecuta las pruebas para verificar la instalación:
```bash
pytest
```
El módulo de análisis ahora forma parte del paquete `tradingbot` y se importa como `tradingbot.analysis`.

## Servicios con Docker
Puedes levantar únicamente las bases de datos con los scripts del
repositorio:

```bash
# TimescaleDB
./bin/start_timescale.sh

# QuestDB
./bin/start_questdb.sh
```

Para el stack completo (API, TimescaleDB, QuestDB, Prometheus y Grafana):

```bash
./bin/start_stack.sh
```

Los archivos de configuración y esquemas se encuentran en `sql/` y
`monitoring/`.

### Dashboards de Grafana
Las plantillas de paneles se encuentran en `monitoring/grafana/dashboards/`.
Al iniciar el stack con `./bin/start_stack.sh` estos dashboards se cargan
automáticamente. Para cargarlos manualmente en una instalación existente,
importa los archivos JSON desde la interfaz de Grafana o copia el directorio
en `/var/lib/grafana/dashboards` y reinicia el servicio.
