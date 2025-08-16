# Monitoring

This project ships a minimal monitoring stack to track bot health and
trading performance.

## FastAPI panel

Run the monitoring panel with:

```bash
uvicorn monitoring.panel:app --reload
```

Available endpoints:

- `GET /metrics` – Prometheus metrics.
- `GET /metrics/summary` – compact JSON snapshot of key metrics.
- `GET /strategies/status` – current strategy states.
- `POST /strategies/{name}/{status}` – update a strategy state.
- `GET /summary` – metrics and strategy states combined.
- `GET /health` – basic liveness probe.
- `GET /dashboards` – list of Grafana dashboards with direct URLs.
- `GET /dashboards/{name}` – redirect to a specific Grafana dashboard.

Static assets are served from `monitoring/static/` for a quick HTML view.

## Grafana dashboards

The monitoring stack ships with ready‑to‑use Grafana and Prometheus
configuration. To launch both services with the provided dashboards and
data source, run:

```bash
docker-compose up -d prometheus grafana
```

Prometheus uses `monitoring/prometheus.yml` to scrape the API and
monitoring panel. Grafana is provisioned from files under
`monitoring/grafana/` which includes:

* `datasources/datasource.yml` – Prometheus data source
* `dashboards/dashboard.yml` – automatic dashboard loading
* `dashboards/core.json` and `dashboards/tradebot.json` – example panels

To build a standalone Grafana image with these files baked in:

```bash
docker build -t tradebot-grafana monitoring/grafana
docker run -p 3000:3000 tradebot-grafana
```

Set `GRAFANA_URL` to point the FastAPI panel to a remote Grafana instance
if it is not running on `localhost:3000`.

Customize the panel by editing `datasources/datasource.yml` or adding new
JSON dashboards under `monitoring/grafana/dashboards/`.

The `core.json` dashboard provides panels for:

- Trading PnL
- End‑to‑end latency (95th percentile)
- Websocket failures per adapter
- Kill‑switch status

## Alerts

Prometheus alerting rules are defined in `monitoring/alerts.yml`,
covering negative PnL, high latencies, websocket failures and the
kill‑switch being triggered.

Load the rules file into Prometheus or Alertmanager to enable basic
notifications.
