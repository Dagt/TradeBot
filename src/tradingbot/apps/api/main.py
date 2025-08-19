# src/tradingbot/apps/api/main.py
from fastapi import FastAPI, Query, HTTPException, Depends, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import time
from starlette.requests import Request
import os
import secrets
import subprocess
import sys
import shlex
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from monitoring.metrics import router as metrics_router
from monitoring.dashboard import router as dashboard_router

from ...storage.timescale import select_recent_fills
from ...utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from ...config import settings

# Persistencia
try:
    from ...storage.timescale import (
        get_engine,
        select_recent_tri_signals,
        select_recent_orders,
        select_order_history,
    )
    _CAN_PG = True
    _ENGINE = get_engine()
except Exception:
    _CAN_PG = False
    _ENGINE = None

security = HTTPBasic()


def _check_basic_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
    user = os.getenv("API_USER", "admin")
    password = os.getenv("API_PASS", "admin")
    correct_user = secrets.compare_digest(credentials.username, user)
    correct_pass = secrets.compare_digest(credentials.password, password)
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )


app = FastAPI(title="TradingBot API", dependencies=[Depends(_check_basic_auth)])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(metrics_router)
app.include_router(dashboard_router)


@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    endpoint = request.url.path
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    return response

@app.get("/health")
def health():
    return {"status": "ok", "db": bool(_CAN_PG)}

@app.get("/logs")
def logs(lines: int = Query(200, ge=1, le=1000)):
    """Return recent lines from the log file."""
    log_file = Path(settings.log_file or "tradingbot.log")
    if not log_file.exists():
        return {"items": [], "warning": "log file not found"}
    try:
        content = log_file.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {"items": [], "warning": "unable to read log file"}
    return {"items": content[-lines:]}

@app.get("/tri/signals")
def tri_signals(limit: int = Query(100, ge=1, le=1000)):
    if not _CAN_PG:
        return {"items": [], "warning": "Timescale/SQLAlchemy no disponible"}
    return {"items": select_recent_tri_signals(_ENGINE, limit=limit)}

@app.get("/orders")
def orders(limit: int = Query(100, ge=1, le=1000)):
    if not _CAN_PG:
        return {"items": [], "warning": "Timescale/SQLAlchemy no disponible"}
    return {"items": select_recent_orders(_ENGINE, limit=limit)}


@app.get("/orders/history")
def orders_history(
    limit: int = Query(100, ge=1, le=1000),
    search: str | None = None,
    symbol: str | None = None,
    status: str | None = None,
):
    if not _CAN_PG:
        return {"items": [], "warning": "Timescale/SQLAlchemy no disponible"}
    return {
        "items": select_order_history(
            _ENGINE, limit=limit, search=search, symbol=symbol, status=status
        )
    }

@app.get("/tri/summary")
def tri_summary(limit: int = Query(200, ge=1, le=5000)):
    if not _CAN_PG:
        return {"summary": {"count": 0, "avg_edge": None, "last_edge": None, "avg_notional": None}}
    rows = select_recent_tri_signals(_ENGINE, limit=limit)
    if not rows:
        return {"summary": {"count": 0, "avg_edge": None, "last_edge": None, "avg_notional": None}}
    count = len(rows)
    avg_edge = sum(r["edge"] for r in rows) / count
    last_edge = rows[0]["edge"]
    avg_notional = sum(float(r["notional_quote"]) for r in rows) / count
    return {"summary": {"count": count, "avg_edge": avg_edge, "last_edge": last_edge, "avg_notional": avg_notional}}

# --- Static dashboard ---
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Servir index.html en "/"
@app.get("/")
def index():
    try:
        html = (_static_dir / "index.html").read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")
    except Exception:
        return {"message": "Sube el dashboard en /static/index.html"}

@app.get("/risk/exposure")
def risk_exposure(venue: str = "binance_spot_testnet"):
    if not _CAN_PG:
        return {"venue": venue, "total_notional": 0.0, "items": []}
    from ...storage.timescale import select_latest_portfolio
    rows = select_latest_portfolio(_ENGINE, venue=venue)
    total = sum(float(r["notional_usd"] or 0.0) for r in rows)
    items = [
        {"symbol": r["symbol"], "position": float(r["position"]), "price": float(r["price"]), "notional_usd": float(r["notional_usd"])}
        for r in rows
    ]
    return {"venue": venue, "total_notional": total, "items": items}

@app.get("/risk/events")
def risk_events(venue: str = "binance_spot_testnet", limit: int = Query(50, ge=1, le=200)):
    if not _CAN_PG:
        return {"venue": venue, "items": []}
    from ...storage.timescale import select_recent_risk_events
    items = select_recent_risk_events(_ENGINE, venue=venue, limit=limit)
    return {"venue": venue, "items": items}


class HaltRequest(BaseModel):
    reason: str


@app.post("/risk/halt")
async def risk_halt(payload: HaltRequest):
    bus = getattr(app.state, "bus", None)
    if bus is None:
        rm = getattr(app.state, "risk_manager", None)
        bus = getattr(rm, "bus", None) if rm else None
    if bus is None:
        raise HTTPException(status_code=500, detail="bus not configured")
    await bus.publish("control:halt", {"reason": payload.reason})
    return {"status": "halt"}


@app.post("/risk/reset")
def risk_reset():
    rm = getattr(app.state, "risk_manager", None)
    if rm is None:
        raise HTTPException(status_code=500, detail="risk manager not configured")
    rm.reset()
    return {"risk": {"enabled": rm.enabled, "last_kill_reason": rm.last_kill_reason}}

@app.get("/pnl/summary")
def pnl_summary(venue: str = "binance_spot_testnet"):
    if not _CAN_PG:
        return {"venue": venue, "items": [], "totals": {"upnl":0,"rpnl":0,"fees":0,"net_pnl":0}}
    from ...storage.timescale import select_pnl_summary
    return select_pnl_summary(_ENGINE, venue=venue)

@app.get("/pnl/timeseries")
def pnl_timeseries(
    venue: str = "binance_spot_testnet",
    symbol: str | None = Query(None),
    bucket: str = Query("1 minute"),
    hours: int = Query(6, ge=1, le=168)   # hasta 7 días
):
    if not _CAN_PG:
        return {"venue": venue, "symbol": symbol, "bucket": bucket, "hours": hours, "points": []}
    from ...storage.timescale import select_pnl_timeseries
    points = select_pnl_timeseries(_ENGINE, venue=venue, symbol=symbol, bucket=bucket, hours=hours)
    return {"venue": venue, "symbol": symbol, "bucket": bucket, "hours": hours, "points": points}

@app.get("/fills/recent")
def fills_recent(
    venue: str = "binance_spot_testnet",
    symbol: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500)
):
    if not _CAN_PG:
        return {"venue": venue, "items": []}
    items = select_recent_fills(_ENGINE, venue=venue, symbol=symbol, limit=limit)
    return {"venue": venue, "symbol": symbol, "items": items}

@app.get("/positions/rebuild_preview")
def positions_rebuild_preview(venue: str = "binance_spot_testnet"):
    """
    No modifica BD; devuelve posiciones calculadas desde fills para comparar con market.positions
    """
    if not _CAN_PG:
        return {"venue": venue, "from_fills": {}, "current": []}
    from ...storage.timescale import rebuild_positions_from_fills, select_pnl_summary
    recalced = rebuild_positions_from_fills(_ENGINE, venue=venue)
    current = select_pnl_summary(_ENGINE, venue=venue)["items"]
    return {"venue": venue, "from_fills": recalced, "current": current}

@app.get("/fills/slippage")
def fills_slippage(
    venue: str = "binance_spot_testnet",
    hours: int = Query(6, ge=1, le=168),
    symbol: str | None = Query(None)
):
    if not _CAN_PG:
        return {"venue": venue, "hours": hours, "symbol": symbol, "global": {}, "buy": {}, "sell": {}}
    from ...storage.timescale import select_slippage
    return select_slippage(_ENGINE, venue=venue, symbol=symbol, hours=hours)


@app.get("/pnl/intraday")
def pnl_intraday(
    venue: str = "binance_spot_testnet",
    symbol: str | None = Query(None)
):
    """Return intraday net PnL for the last 24h."""
    if not _CAN_PG:
        return {"venue": venue, "symbol": symbol, "net": 0.0, "points": []}
    from ...storage.timescale import select_pnl_timeseries
    points = select_pnl_timeseries(
        _ENGINE, venue=venue, symbol=symbol, bucket="1 hour", hours=24
    )
    net = sum(p.get("net", 0.0) for p in points)
    return {"venue": venue, "symbol": symbol, "net": net, "points": points}

@app.get("/oco/active")
def oco_active(venue: str, symbols: str):
    """
    symbols: CSV "BTC/USDT,ETH/USDT"
    """
    if not _CAN_PG:
        return {}
    from ...storage.timescale import load_active_oco_by_symbols
    lst = [s.strip().upper().replace("-", "/") for s in symbols.split(",") if s.strip()]
    return load_active_oco_by_symbols(_ENGINE, venue=venue, symbols=lst)


# --- Strategy control endpoints -------------------------------------------------
_strategies_state: dict[str, str] = {}
_strategy_params: dict[str, dict] = {}
_funding: dict[str, float] = {}
_basis: dict[str, float] = {}


@app.post("/strategies/{name}/start")
def start_strategy(name: str):
    """Mark a strategy as running."""

    from ...strategies import STRATEGIES

    if name not in STRATEGIES:
        raise HTTPException(status_code=404, detail="unknown strategy")
    _strategies_state[name] = "running"
    return {"strategy": name, "status": "running"}


@app.post("/strategies/{name}/stop")
def stop_strategy(name: str):
    """Mark a strategy as stopped."""

    from ...strategies import STRATEGIES

    if name not in STRATEGIES:
        raise HTTPException(status_code=404, detail="unknown strategy")
    _strategies_state[name] = "stopped"
    return {"strategy": name, "status": "stopped"}


@app.get("/strategies/status")
def strategies_status():
    """Return the status of all known strategies."""

    return {"strategies": _strategies_state}


# --- Funding and basis endpoints ------------------------------------------------


@app.get("/funding")
def get_funding():
    """Return latest funding rates by symbol."""

    return {"funding": _funding}


@app.post("/funding")
def set_funding(data: dict[str, float]):
    """Update funding rates for one or more symbols."""

    _funding.update(data)
    return {"funding": _funding}


@app.get("/basis")
def get_basis():
    """Return latest basis values by symbol."""

    return {"basis": _basis}


@app.post("/basis")
def set_basis(data: dict[str, float]):
    """Update basis values for one or more symbols."""

    _basis.update(data)
    return {"basis": _basis}


# --- Strategy parameters --------------------------------------------------------


@app.get("/strategies/{name}/params")
def get_strategy_params(name: str):
    """Return stored parameters for a strategy."""

    return {"strategy": name, "params": _strategy_params.get(name, {})}


@app.post("/strategies/{name}/params")
def update_strategy_params(name: str, params: dict):
    """Persist parameters for a strategy."""

    _strategy_params[name] = params
    return {"strategy": name, "params": params}


# --- CLI runner ------------------------------------------------------------------


class CLIRequest(BaseModel):
    """Payload for running a command via the CLI."""

    command: str


@app.post("/cli/run")
def run_cli(req: CLIRequest):
    """Execute a TradingBot CLI command and return its output.

    The endpoint runs ``python -m tradingbot.cli`` with the arguments supplied
    in ``command``.  Output from ``stdout`` and ``stderr`` is captured and
    returned to the caller so it can be displayed in the dashboard.
    """

    args = shlex.split(req.command)
    cmd = [sys.executable, "-m", "tradingbot.cli", *args]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return {
            "command": req.command,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except Exception as exc:  # pragma: no cover - subprocess failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc

