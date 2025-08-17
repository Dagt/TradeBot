"""Monitoring panel exposing metrics and strategy state via FastAPI."""

import os
import json
from pathlib import Path

import httpx
import yaml
import sentry_sdk
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .metrics import router as metrics_router, metrics_summary
from .strategies import (
    router as strategies_router,
    strategies_status,
)

config_path = Path(__file__).with_name("sentry.yml")
if config_path.exists():
    with config_path.open() as f:
        config = yaml.safe_load(f)
    sentry_sdk.init(integrations=[FastApiIntegration()], **config)

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)
app.include_router(strategies_router)


GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

dashboards_dir = Path(__file__).parent / "grafana" / "dashboards"
GRAFANA_DASHBOARDS: dict[str, str] = {}
for path in dashboards_dir.glob("*.json"):
    try:
        with path.open() as f:
            data = json.load(f)
        uid = data.get("uid")
        if uid:
            GRAFANA_DASHBOARDS[path.stem] = uid
    except Exception:
        continue


@app.get("/dashboards")
def dashboards() -> dict[str, str]:
    """Return available Grafana dashboards with direct URLs."""

    return {name: f"{GRAFANA_URL}/d/{uid}" for name, uid in GRAFANA_DASHBOARDS.items()}


@app.get("/dashboards/{name}")
def dashboard_link(name: str) -> RedirectResponse:
    """Redirect to a Grafana dashboard by name."""

    uid = GRAFANA_DASHBOARDS.get(name)
    if not uid:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return RedirectResponse(f"{GRAFANA_URL}/d/{uid}")


def fetch_alerts() -> list[dict]:
    """Return current alerts from Prometheus."""

    try:
        resp = httpx.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=5.0)
        resp.raise_for_status()
        return resp.json().get("data", {}).get("alerts", [])
    except httpx.HTTPError:
        return []


@app.get("/alerts")
def alerts() -> dict[str, bool | list[dict]]:
    """Expose current alert state with flags for key alerts."""

    current = fetch_alerts()
    active = {a.get("labels", {}).get("alertname") for a in current}
    return {
        "risk_events": "RiskEvents" in active,
        "kill_switch_active": "KillSwitchActive" in active,
        "ws_disconnects": "WebsocketDisconnects" in active,
        "alerts": current,
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Basic health check endpoint."""

    return {"status": "ok"}


@app.get("/summary")
def summary() -> dict:
    """Return key metrics and current strategy states."""

    return {
        "metrics": metrics_summary(),
        "strategies": strategies_status()["strategies"],
        "alerts": fetch_alerts(),
    }

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
