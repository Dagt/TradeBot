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
