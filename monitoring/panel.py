"""Monitoring panel exposing metrics and strategy state via FastAPI."""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import yaml
import sentry_sdk
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
GRAFANA_DASHBOARDS = {
    "core": "core",
    "tradebot": "tradebot",
}


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
    }

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
