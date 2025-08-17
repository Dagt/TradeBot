"""Monitoring panel exposing metrics and strategy state via FastAPI."""

import os
import json
from pathlib import Path
import logging
import asyncio

import httpx
import yaml
import sentry_sdk
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .metrics import (
    router as metrics_router,
    metrics_summary,
    STRATEGY_ACTIONS,
    TRADING_PNL,
    OPEN_POSITIONS,
    RISK_EVENTS,
    KILL_SWITCH_ACTIVE,
    PROCESS_CPU,
    PROCESS_MEMORY,
    PROCESS_UPTIME,
    update_process_metrics,
)
from .strategies import strategies_status, set_strategy_status

config_path = Path(__file__).with_name("sentry.yml")
if config_path.exists():
    with config_path.open() as f:
        config = yaml.safe_load(f)
    sentry_sdk.init(integrations=[FastApiIntegration()], **config)

logger = logging.getLogger(__name__)

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)

_strategy_params: dict[str, dict] = {}


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

    return {
        name: f"{GRAFANA_URL}/d/{uid}"
        for name, uid in GRAFANA_DASHBOARDS.items()
    }


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

    update_process_metrics()
    system = {
        "cpu_percent": PROCESS_CPU._value.get(),
        "memory_bytes": PROCESS_MEMORY._value.get(),
        "process_uptime_seconds": PROCESS_UPTIME._value.get(),
    }

    return {
        "metrics": metrics_summary(),
        "system": system,
        "strategies": strategies_status()["strategies"],
        "alerts": fetch_alerts(),
    }


@app.get("/pnl")
def pnl() -> dict:
    """Return current trading PnL."""

    return {"pnl": TRADING_PNL._value.get()}


@app.get("/pnl/summary")
def pnl_summary() -> dict:
    """Return summary PnL (alias of /pnl)."""

    return {"pnl": TRADING_PNL._value.get()}


@app.get("/positions")
def positions() -> dict:
    """Return current open positions by symbol."""

    pos = {
        sample.labels["symbol"]: sample.value
        for metric in OPEN_POSITIONS.collect()
        for sample in metric.samples
        if sample.name == "open_position"
    }
    return {"positions": pos}


@app.get("/risk/exposure")
def risk_exposure() -> dict:
    """Return current risk exposure per symbol."""

    pos = {
        sample.labels["symbol"]: sample.value
        for metric in OPEN_POSITIONS.collect()
        for sample in metric.samples
        if sample.name == "open_position"
    }
    return {"exposure": pos}


@app.get("/risk/events")
def risk_events() -> dict:
    """Return total risk management events."""

    total = sum(
        sample.value
        for metric in RISK_EVENTS.collect()
        for sample in metric.samples
        if sample.name.endswith("_total")
    )
    return {"events": total}


@app.get("/kill-switch")
def kill_switch() -> dict:
    """Return whether the kill switch is active."""

    return {"kill_switch_active": bool(KILL_SWITCH_ACTIVE._value.get())}


@app.post("/strategies/{name}/enable")
def enable_strategy(name: str, params: dict | None = None) -> dict:
    """Enable a strategy and optionally update its parameters."""

    logger.info("Enabling strategy %s", name)
    res = set_strategy_status(name, "running")
    if params:
        _strategy_params[name] = params
    STRATEGY_ACTIONS.labels(strategy=name, action="enable").inc()
    res["params"] = _strategy_params.get(name, {})
    return res


@app.post("/strategies/{name}/disable")
def disable_strategy(name: str) -> dict:
    """Disable a strategy."""

    logger.info("Disabling strategy %s", name)
    res = set_strategy_status(name, "stopped")
    STRATEGY_ACTIONS.labels(strategy=name, action="disable").inc()
    return res


@app.post("/strategies/{name}/params")
def update_strategy_params(name: str, params: dict) -> dict:
    """Update parameters for a strategy."""

    logger.info("Updating params for %s: %s", name, params)
    _strategy_params[name] = params
    STRATEGY_ACTIONS.labels(strategy=name, action="params").inc()
    return {"strategy": name, "params": params}


@app.get("/strategies/status")
def get_strategies_status() -> dict:
    """Return the status of all strategies."""

    return strategies_status()


@app.websocket("/ws")
async def ws_updates(ws: WebSocket) -> None:
    """Send periodic metric, PnL and risk updates over a WebSocket."""

    await ws.accept()
    try:
        while True:
            exposure = {
                sample.labels["symbol"]: sample.value
                for metric in OPEN_POSITIONS.collect()
                for sample in metric.samples
                if sample.name == "open_position"
            }
            events = sum(
                sample.value
                for metric in RISK_EVENTS.collect()
                for sample in metric.samples
                if sample.name.endswith("_total")
            )
            payload = {
                "metrics": metrics_summary(),
                "pnl": {"pnl": TRADING_PNL._value.get()},
                "risk": {"exposure": exposure, "events": events},
            }
            await ws.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
