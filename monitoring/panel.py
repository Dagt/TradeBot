"""Monitoring panel exposing metrics and strategy state via FastAPI."""

import os
import json
from pathlib import Path
import logging
import asyncio
import shlex
import importlib
import inspect

import httpx
import yaml
import sentry_sdk
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentry_sdk.integrations.fastapi import FastApiIntegration
from tradingbot.exchanges import SUPPORTED_EXCHANGES

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
from .strategies import (
    strategies_status,
    set_strategy_status,
    register_strategy,
    update_strategy_params as set_strategy_params,
    get_strategy_params,
    available_strategies,
)
from .alerts import evaluate_alerts

config_path = Path(__file__).with_name("sentry.yml")
if config_path.exists():
    with config_path.open() as f:
        config = yaml.safe_load(f)
    sentry_sdk.init(integrations=[FastApiIntegration()], **config)

logger = logging.getLogger(__name__)

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)


# ---------------------------------------------------------------------------
# Strategy discovery
# ---------------------------------------------------------------------------
# On startup, automatically register all strategy modules found under
# ``tradingbot/strategies`` so they become available via the API.

STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "src" / "tradingbot" / "strategies"


@app.on_event("startup")
def discover_strategies() -> None:
    for path in STRATEGIES_DIR.glob("*.py"):
        if path.name in {"base.py", "__init__.py"}:
            continue
        register_strategy(path.stem)


@app.get("/strategies")
def list_strategies() -> dict:
    """Return the names of all registered strategies."""

    return available_strategies()


@app.get("/strategies/{name}/schema")
def strategy_schema(name: str) -> dict:
    """Inspect a strategy's ``__init__`` signature.

    Returns parameter names, annotated types and default values so the
    frontend can render a form for configuring the strategy.
    """

    try:
        module = importlib.import_module(f"tradingbot.strategies.{name}")
    except ModuleNotFoundError:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy_cls = None
    for attr in dir(module):
        obj = getattr(module, attr)
        if inspect.isclass(obj) and getattr(obj, "name", None) == name:
            strategy_cls = obj
            break
    if strategy_cls is None:
        raise HTTPException(status_code=404, detail="Strategy class not found")

    sig = inspect.signature(strategy_cls.__init__)
    params = []
    for p in list(sig.parameters.values())[1:]:  # skip ``self``
        if p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if p.annotation is inspect._empty:
            annotation = "Any"
        elif isinstance(p.annotation, type):
            annotation = p.annotation.__name__
        else:
            annotation = str(p.annotation)
        default = None if p.default is inspect._empty else p.default
        params.append({"name": p.name, "type": annotation, "default": default})
    return {"strategy": name, "params": params}

# ---------------------------------------------------------------------------
# Configuration storage
# ---------------------------------------------------------------------------
# ``_config`` holds the current bot settings as provided via the ``/config``
# endpoint.  ``_bot_process`` stores the subprocess handle for an actively
# running strategy so it can later be terminated.

_config: dict[str, object] = {
    "strategy": None,
    "pairs": [],
    "notional": None,
    "params": {},
    "venue": "binance_spot",
    "risk_pct": 0.0,
    "leverage": 1,
    "testnet": True,
    "dry_run": False,
}
_bot_process: asyncio.subprocess.Process | None = None

# Persist configuration to a JSON file so values survive restarts.  The file
# lives alongside this module to avoid requiring any additional directory
# structure.
CONFIG_FILE = Path(__file__).with_name("config.json")

# Load existing configuration if available.  Errors are logged but ignored so a
# corrupt config does not prevent the panel from starting.
if CONFIG_FILE.exists():
    try:
        with CONFIG_FILE.open() as fh:
            _config.update(json.load(fh) or {})
    except Exception:  # pragma: no cover - extremely unlikely
        logger.warning("Failed to load config file", exc_info=True)


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


# ---------------------------------------------------------------------------
# Configuration endpoints
# ---------------------------------------------------------------------------


class BotConfig(BaseModel):
    """Pydantic model for bot configuration updates."""

    strategy: str | None = None
    pairs: list[str] | None = None
    notional: float | None = None
    params: dict | None = None
    venue: str | None = None
    risk_pct: float | None = None
    leverage: int | None = None
    testnet: bool | None = None
    dry_run: bool | None = None


@app.get("/config")
def get_config() -> dict:
    """Return the current bot configuration."""

    return {"config": _config}


@app.post("/config")
def update_config(cfg: BotConfig) -> dict:
    """Update the in-memory bot configuration and persist it to disk."""

    data = cfg.dict(exclude_unset=True)

    # Remove ``None`` values so partial updates work as expected and run a few
    # basic validations.  FastAPI/Pydantic already ensures types but we enforce
    # simple domain rules such as positive notionals.
    for key, value in list(data.items()):
        if value is None:
            data.pop(key)
            continue
        if key == "notional" and value <= 0:
            raise HTTPException(status_code=400, detail="notional must be positive")
        if key == "risk_pct" and value < 0:
            raise HTTPException(status_code=400, detail=f"{key} must be non-negative")
        if key == "leverage" and value <= 0:
            raise HTTPException(status_code=400, detail="leverage must be positive")
        if key == "pairs":
            if not isinstance(value, list) or not all(isinstance(p, str) and p for p in value):
                raise HTTPException(status_code=400, detail="pairs must be a list of symbols")
        if key == "strategy" and not value:
            raise HTTPException(status_code=400, detail="strategy must be provided")
        if key == "venue" and value not in SUPPORTED_EXCHANGES:
            raise HTTPException(status_code=400, detail="invalid venue")

    _config.update(data)

    # Persist configuration so the state survives restarts.
    try:
        with CONFIG_FILE.open("w") as fh:
            json.dump(_config, fh)
    except OSError as exc:  # pragma: no cover - unlikely in tests
        logger.error("Failed to persist config", exc_info=True)
        raise HTTPException(status_code=500, detail="failed to persist config") from exc

    strategy = _config.get("strategy")
    params = _config.get("params") or {}
    if strategy:
        register_strategy(strategy, params if params else None)
        if params:
            set_strategy_params(strategy, params)
    return {"config": _config}


# ---------------------------------------------------------------------------
# Bot lifecycle endpoints
# ---------------------------------------------------------------------------


@app.post("/bot/start")
async def bot_start() -> dict:
    """Launch the configured strategy via ``tradingbot.cli``."""

    global _bot_process
    if _bot_process and _bot_process.returncode is None:
        raise HTTPException(status_code=400, detail="Bot already running")
    strategy = _config.get("strategy")
    if not strategy:
        raise HTTPException(status_code=400, detail="Strategy not configured")
    args = [
        "python",
        "-m",
        "tradingbot.cli",
        "run-bot",
        "--strategy",
        strategy,
    ]
    for pair in _config.get("pairs") or []:
        args.extend(["--symbol", pair])
    if _config.get("notional") is not None:
        args.extend(["--notional", str(_config["notional"])])
    if _config.get("venue"):
        args.extend(["--venue", str(_config["venue"])])
    if _config.get("risk_pct") is not None:
        args.extend(["--risk-pct", str(_config["risk_pct"])])
    if _config.get("leverage") is not None:
        args.extend(["--leverage", str(_config["leverage"])])
    testnet = _config.get("testnet")
    if testnet is not None:
        args.append("--testnet" if testnet else "--no-testnet")
    dry_run = _config.get("dry_run")
    if dry_run is not None:
        args.append("--dry-run" if dry_run else "--no-dry-run")
    _bot_process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    set_strategy_status(strategy, "running")
    return {"status": "started", "pid": _bot_process.pid}


@app.post("/bot/stop")
async def bot_stop() -> dict:
    """Terminate a running strategy process."""

    global _bot_process
    if not _bot_process or _bot_process.returncode is not None:
        raise HTTPException(status_code=400, detail="Bot is not running")
    _bot_process.terminate()
    await _bot_process.wait()
    rc = _bot_process.returncode
    _bot_process = None
    strategy = _config.get("strategy")
    if strategy:
        set_strategy_status(strategy, "stopped")
    return {"status": "stopped", "returncode": rc}


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
    register_strategy(name, params)
    if params:
        set_strategy_params(name, params)
    res = set_strategy_status(name, "running")
    STRATEGY_ACTIONS.labels(strategy=name, action="enable").inc()
    res["params"] = get_strategy_params(name)["params"]
    return res


@app.post("/strategies/{name}/disable")
def disable_strategy(name: str) -> dict:
    """Disable a strategy."""

    logger.info("Disabling strategy %s", name)
    res = set_strategy_status(name, "stopped")
    STRATEGY_ACTIONS.labels(strategy=name, action="disable").inc()
    res["params"] = get_strategy_params(name).get("params", {})
    return res


@app.post("/strategies/{name}/params")
def update_strategy_params_endpoint(name: str, params: dict) -> dict:
    """Update parameters for a strategy."""

    logger.info("Updating params for %s: %s", name, params)
    res = set_strategy_params(name, params)
    STRATEGY_ACTIONS.labels(strategy=name, action="params").inc()
    return res


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
        "command": cmd.command,
        "stdout": out.decode(),
        "stderr": err.decode(),
        "returncode": proc.returncode,
        "success": proc.returncode == 0,
    }


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
