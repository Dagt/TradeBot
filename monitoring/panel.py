"""Monitoring panel exposing metrics and strategy state via FastAPI."""

from fastapi import FastAPI
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
