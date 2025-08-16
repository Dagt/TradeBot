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

Static assets are served from `monitoring/static/` for a quick HTML view.

## Grafana dashboards

Provisioning files live under `monitoring/grafana/`. A new dashboard
`core.json` provides panels for:

- Trading PnL
- End‑to‑end latency (95th percentile)
- Websocket failures per adapter
- Kill‑switch status

Place the JSON files in Grafana's dashboards directory or mount the
folder when running the official container.

## Alerts

Prometheus alerting rules are defined in `monitoring/alerts.yml`,
covering negative PnL, high latencies, websocket failures and the
kill‑switch being triggered.

Load the rules file into Prometheus or Alertmanager to enable basic
notifications.
