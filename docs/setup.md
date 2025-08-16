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

## Dashboards de Grafana
Los dashboards listos para usar están en `monitoring/grafana/dashboards/`.
Al ejecutar `./bin/start_stack.sh` se provisionan automáticamente.
Para una instancia de Grafana existente copia `tradebot.json` al
directorio de dashboards (`/var/lib/grafana/dashboards`) y `dashboard.yml`
al directorio de provisioning (`/etc/grafana/provisioning/dashboards/`).
