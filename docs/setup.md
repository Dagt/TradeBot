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

Una vez instalado, puedes explorar funciones avanzadas descritas en
[extra_features.md](extra_features.md), como estrategias de arbitraje y
el panel web con ejecución de comandos.

## Descarga de datos
Puedes descargar históricos de barras, trades o snapshots de libro L2 con
`bin/download_history.py`.  Los datos se normalizan y pueden guardarse en
archivos CSV dentro de `db/` o directamente en TimescaleDB/QuestDB:

```bash
# Descargar trades a CSV
./bin/download_history.py trades binance BTC/USDT 2024-01-01 2024-01-02 --backend csv

# Guardar snapshots L2 en TimescaleDB
./bin/download_history.py l2 binance BTC/USDT --snapshots 5 --backend timescale
```

La persistencia se realiza mediante utilidades de `tradingbot.data.ingestion`.

### Claves de API para conectores externos

Algunos proveedores como Kaiko y CoinAPI requieren claves de autenticación.
Establece las variables de entorno `KAIKO_API_KEY` y `COINAPI_KEY` antes de
invocar los conectores o scripts que los utilicen:

```bash
export KAIKO_API_KEY="tu_clave_kaiko"
export COINAPI_KEY="tu_clave_coinapi"
```

Los conectores `KaikoConnector` y `CoinAPIConnector` ofrecen métodos
``fetch_trades`` y ``fetch_order_book`` que pueden emplearse junto a las
utilidades de `ingestion` para descargar históricos y persistirlos.

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
