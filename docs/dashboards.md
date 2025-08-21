# Dashboards de Monitoreo

TradeBot incluye paneles de Grafana para visualizar el estado del bot y los
resultados de las operaciones. Los archivos de cada dashboard se encuentran
en `monitoring/grafana/dashboards`.

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
