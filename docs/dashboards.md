# Dashboards de Monitoreo

TradeBot incluye paneles de Grafana para visualizar el estado del bot y los
resultados de las operaciones. Los archivos de cada dashboard se encuentran
en `monitoring/grafana/dashboards`.

Por defecto, la API expone sus métricas en `http://localhost:8000/metrics`.
Si se ejecutan varios bots, cada uno tomará el primer puerto libre del rango
`8001-8010`, quedando accesibles desde `http://localhost:<puerto>/metrics`.

## Variables de consulta
Para los paneles que acceden a datos de mercado desde PostgreSQL se añadieron
variables que permiten filtrar las consultas:

- **$symbol**: símbolo o par de trading a visualizar.
- **$start**: timestamp inicial del rango a consultar.
- **$end**: timestamp final del rango.

Estas variables pueden emplearse en las cláusulas `WHERE` de las consultas a
`market.bars` o `market.trades` para limitar los resultados por símbolo y
periodo.

## Core Metrics (`core.json`)
Muestra métricas básicas del proceso:
- **Trading PnL**: resultado acumulado.
- **E2E latency (95th)**: latencia extremo a extremo del sistema.
- **WS failures**: cantidad de fallos en conexiones WebSocket.
- **Kill switch active**: indicador del interruptor de emergencia.
- **CPU Usage** y **Memory Usage**: consumo de recursos.
- **Process Uptime**: tiempo que lleva corriendo el bot.

## PnL & Positions (`pnl_positions.json`)
Enfocado en resultados y exposición:
- **Trading PnL**.
- **Kill switch active**.
- **Open Positions**: gráfico de posiciones abiertas por símbolo.

## TradeBot Metrics (`tradebot.json`)
Vista general del desempeño del bot:
- **API latency (95th)**.
- **WS failures** por adaptador.
- **Order fills**: órdenes ejecutadas.
- **Risk events (5m)**: eventos de riesgo recientes.
- **E2E latency (95th)**.
- **Strategy states**: estado reportado por cada estrategia.

## Carga de datos históricos
Para completar los paneles con datos antiguos se puede usar el comando
`backfill` indicando un rango explícito:

```bash
python -m tradingbot.cli backfill --symbols BTC/USDT \
    --start 2023-01-01T00:00:00Z --end 2023-01-02T00:00:00Z
```

En el dashboard de "Datos históricos" se incluyen los campos **Inicio** y
**Fin** para facilitar esta tarea. Al especificar ambos valores se usará ese
rango, ocultando la opción de **Días**. Si se dejan vacíos, el comando empleará
el número de días indicado.

Las opciones `--start` y `--end` aceptan fechas en formato ISO.
