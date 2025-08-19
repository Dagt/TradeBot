"""Monitoring panel exposing metrics and strategy state via FastAPI."""

import os
import json
from pathlib import Path
import logging
import asyncio
import shlex

import httpx
import yaml
import sentry_sdk
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .metrics import (
    router as metrics_router,
    metrics_summary,
    STRATEGY_ACTIONS,
    TRADING_PNL,
    OPEN_POSITIONS,
    KILL_SWITCH_ACTIVE,
    PROCESS_CPU,
    PROCESS_MEMORY,
    PROCESS_UPTIME,
    update_process_metrics,
)
from .strategies import strategies_status, set_strategy_status
from .alerts import evaluate_alerts

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
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_USER = os.getenv("API_USER", "admin")
API_PASS = os.getenv("API_PASS", "admin")

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
    """Return current alerts evaluated by :mod:`monitoring.alerts`."""

    return evaluate_alerts()


async def fetch_risk() -> dict:
    """Return risk exposure and recent events from the trading API."""

    exposure: dict = {}
    events: dict = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{API_URL}/risk/exposure")
            resp.raise_for_status()
            exposure = resp.json()
        except httpx.HTTPError:
            exposure = {}
        try:
            resp = await client.get(f"{API_URL}/risk/events?limit=20")
            resp.raise_for_status()
            events = resp.json()
        except httpx.HTTPError:
            events = {}
    return {
        "exposure": exposure.get("exposure"),
        "events": events.get("events", []),
    }


async def fetch_orders() -> list[dict]:
    """Return open orders from the trading API."""

    orders: list[dict] = []
    async with httpx.AsyncClient(timeout=5.0, auth=(API_USER, API_PASS)) as client:
        try:
            resp = await client.get(f"{API_URL}/orders?limit=100")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            data = {}
    items = data.get("items", [])
    for o in items:
        status = (o.get("status") or "").lower()
        if status not in {"filled", "canceled", "rejected"}:
            orders.append(
                {
                    "id": o.get("ext_order_id") or o.get("id"),
                    "symbol": o.get("symbol"),
                    "side": o.get("side"),
                    "status": o.get("status"),
                }
            )
    return orders


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


@app.websocket("/ws/summary")
async def ws_summary(ws: WebSocket) -> None:
    """Stream metrics, PnL and risk data over a WebSocket."""

    await ws.accept()
    try:
        while True:
            update_process_metrics()
            payload = {
                "metrics": metrics_summary(),
                "pnl": {"pnl": TRADING_PNL._value.get()},
                "risk": await fetch_risk(),
            }
            await ws.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


@app.get("/pnl")
def pnl() -> dict:
    """Return current trading PnL."""

    return {"pnl": TRADING_PNL._value.get()}


@app.get("/orders")
async def orders() -> dict:
    """Return currently open orders."""

    return {"orders": await fetch_orders()}


class TradeRequest(BaseModel):
    """Payload for placing a manual trade from the dashboard."""

    symbol: str
    side: str
    qty: float
    price: float | None = None


@app.post("/trade")
async def place_trade(order: TradeRequest) -> dict:
    """Forward a trade request to the trading API."""

    payload = {"symbol": order.symbol, "side": order.side, "qty": order.qty}
    if order.price is not None:
        payload["price"] = order.price
    async with httpx.AsyncClient(auth=(API_USER, API_PASS), timeout=5.0) as client:
        try:
            resp = await client.post(f"{API_URL}/orders", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=500, detail=f"trade failed: {exc}")
    STRATEGY_ACTIONS.labels(strategy="manual", action="trade").inc()
    return {"order": data}


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


class CLICommand(BaseModel):
    """Payload for executing a CLI command from the dashboard."""

    command: str


@app.post("/run-cli")
async def run_cli(cmd: CLICommand) -> dict:
    """Execute a ``tradingbot.cli`` command and return its output."""

    args = shlex.split(cmd.command)
    proc = await asyncio.create_subprocess_exec(
        "python",
        "-m",
        "tradingbot.cli",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    return {
        "stdout": out.decode(),
        "stderr": err.decode(),
        "returncode": proc.returncode,
    }


static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
