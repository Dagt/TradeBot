# Monitoring

This project ships a minimal monitoring stack to track bot health and
trading performance.

## FastAPI panel

Run the monitoring panel with:

```bash
uvicorn monitoring.panel:app --reload
```

The API backing the panel uses HTTP Basic authentication. Configure
credentials via the `API_USER` and `API_PASS` environment variables
(default `admin` / `admin`) and supply them when querying the API.

Available endpoints:

- `GET /metrics` – Prometheus metrics.
- `GET /metrics/summary` – compact JSON snapshot of key metrics.
- `GET /pnl` – current trading PnL.
- `GET /positions` – open positions by symbol.
- `GET /kill-switch` – kill switch active flag.
- `GET /strategies/status` – current strategy states.
- `POST /strategies/{name}/{status}` – update a strategy state.
- `GET /summary` – metrics and strategy states combined.
- `GET /health` – basic liveness probe.
- `GET /alerts` – current firing and pending alerts with risk flags.
- `GET /dashboards` – list of Grafana dashboards with direct URLs.
- `GET /dashboards/{name}` – redirect to a specific Grafana dashboard.

Static assets are served from `monitoring/static/` for a quick HTML view.

### Plotly dashboard

The API also exposes a lightweight dashboard at `/dashboard` which polls
`/metrics/summary` every few seconds and renders basic graphs using
Plotly. Launch the API and open the dashboard in a browser:

```bash
uvicorn tradingbot.apps.api.main:app --reload
# then visit http://localhost:8000/dashboard (use API_USER/API_PASS credentials)
```

## Grafana dashboards

The monitoring stack ships with ready‑to‑use Grafana, Prometheus and
Alertmanager configuration. To launch the services with the provided
dashboards and data source, run:

```bash
docker-compose up -d prometheus alertmanager grafana
```

Prometheus uses `monitoring/prometheus.yml` to scrape the API and
monitoring panel. Grafana is provisioned from files under
`monitoring/grafana/`. Any JSON file dropped into
`monitoring/grafana/dashboards/` is automatically exposed through the
panel's `/dashboards` endpoint. The directory currently includes:

* `datasources/datasource.yml` – Prometheus data source
* `dashboards/dashboard.yml` – automatic dashboard loading
* `dashboards/core.json` and `dashboards/tradebot.json` – example panels
* `dashboards/pnl_positions.json` – PnL, kill‑switch and open positions

To build a standalone Grafana image with these files baked in:

```bash
docker build -t tradebot-grafana monitoring/grafana
docker run -p 3000:3000 tradebot-grafana
```

Set `GRAFANA_URL` and `PROMETHEUS_URL` to point the FastAPI panel to
remote instances if they are not running on `localhost:3000` and
`localhost:9090` respectively.

Customize the panel by editing `datasources/datasource.yml` or adding new
JSON dashboards under `monitoring/grafana/dashboards/`.

The `core.json` dashboard provides panels for:

- Trading PnL
- End‑to‑end latency (95th percentile)
- Websocket failures per adapter
- Kill‑switch status

## Alerts

Prometheus alerting rules are defined in `monitoring/alerts.yml`,
covering negative PnL, high latencies, websocket issues, risk events and
the kill‑switch being triggered. Prometheus loads this file and forwards
firing alerts to Alertmanager, which is configured via
`monitoring/alertmanager.yml`.

Example rules for risk monitoring:

```yaml
- alert: RiskEvents
  expr: increase(risk_events_total[5m]) > 0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: Risk management events detected
    description: Risk triggers fired in the last 5m

- alert: KillSwitchActive
  expr: kill_switch_active > 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: Kill switch engaged
    description: Trading halted via kill switch for over a minute

- alert: WebsocketDisconnects
  expr: increase(ws_failures_total[5m]) > 0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: Websocket disconnections detected
    description: Websocket clients disconnected in the last 5m
```

Edit `alertmanager.yml` to integrate with your preferred notification
service (Slack, email, webhooks, …). After changes, restart the stack so
both Prometheus and Alertmanager pick up the updates:

```bash
docker-compose up -d --force-recreate prometheus alertmanager
```

The monitoring panel exposes active alerts at `GET /alerts` and also
includes them in `GET /summary`.
