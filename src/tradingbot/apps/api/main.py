# src/tradingbot/apps/api/main.py
from fastapi import FastAPI, Query, HTTPException, Depends, status, Response
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
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
import asyncio
import signal
import uuid
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

try:  # pragma: no cover - ccxt is optional
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - if ccxt is missing
    ccxt = None

from monitoring.metrics import router as metrics_router
from monitoring.dashboard import router as dashboard_router

from ...storage.timescale import select_recent_fills
from ...utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from ...config import settings
from ...cli.main import get_adapter_class, get_supported_kinds
from ...exchanges import SUPPORTED_EXCHANGES
from ...core.symbols import normalize as normalize_symbol

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


@app.get("/venues")
def list_venues():
    """Return available venues."""
    return sorted(SUPPORTED_EXCHANGES)


@app.get("/ccxt/exchanges")
def ccxt_exchanges(include_testnet: bool = Query(True)):
    """Return exchanges supported by ``ccxt``.

    The list includes WebSocket-only variants (``*_ws``) which are limited to
    streaming and do not support trading or historical backfills. Testnet
    variants (``*_testnet``) are included by default; pass ``include_testnet=False``
    to return only live exchanges.
    """
    if ccxt is None:
        return []
    available = getattr(ccxt, "exchanges", [])
    live: list[str] = []
    for key, info in SUPPORTED_EXCHANGES.items():
        if info["ccxt"] in available:
            live.append(key)
            if include_testnet:
                live.append(f"{key}_testnet")
    return live


@app.get("/venues/{name}/kinds")
def venue_kinds(name: str):
    """Return available streaming kinds for the given venue."""

    cls = get_adapter_class(name)
    if cls is None:
        raise HTTPException(status_code=404, detail="venue not found")
    return {"kinds": get_supported_kinds(cls)}

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
    symbol = normalize_symbol(symbol) if symbol else None
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

# Servir stats.html en "/stats"
@app.get("/stats")
def stats_page():
    try:
        html = (_static_dir / "stats.html").read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")
    except Exception:
        return {"message": "Sube el dashboard en /static/stats.html"}


# Servir data.html en "/data"
@app.get("/data")
def data_page():
    try:
        html = (_static_dir / "data.html").read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")
    except Exception:
        return {"message": "Sube el dashboard en /static/data.html"}


# Servir backtest.html en "/backtest"
@app.get("/backtest")
def backtest_page():
    try:
        html = (_static_dir / "backtest.html").read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")
    except Exception:
        return {"message": "Sube el dashboard en /static/backtest.html"}

# Compatibilidad: redirige "/monitor" a "/stats"
@app.get("/monitor")
def monitor_redirect():
    return RedirectResponse("/stats")

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
def pnl_summary(venue: str = "binance_spot_testnet", symbol: str | None = Query(None)):
    if not _CAN_PG:
        return {"venue": venue, "symbol": symbol, "items": [], "totals": {"upnl":0,"rpnl":0,"fees":0,"net_pnl":0}}
    from ...storage.timescale import select_pnl_summary
    symbol = normalize_symbol(symbol) if symbol else None
    return select_pnl_summary(_ENGINE, venue=venue, symbol=symbol)

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
    symbol = normalize_symbol(symbol) if symbol else None
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
    symbol = normalize_symbol(symbol) if symbol else None
    items = select_recent_fills(_ENGINE, venue=venue, symbol=symbol, limit=limit)
    return {"venue": venue, "symbol": symbol, "items": items}

@app.get("/fills/summary")
def fills_summary(
    venue: str = "binance_spot_testnet",
    symbol: str | None = Query(None),
    strategy: str | None = Query(None),
):
    if not _CAN_PG:
        return {"venue": venue, "symbol": symbol, "strategy": strategy, "items": []}
    from ...storage.timescale import select_fills_summary
    symbol = normalize_symbol(symbol) if symbol else None
    items = select_fills_summary(_ENGINE, venue=venue, symbol=symbol, strategy=strategy)
    return {"venue": venue, "symbol": symbol, "strategy": strategy, "items": items}

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
    symbol = normalize_symbol(symbol) if symbol else None
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
    symbol = normalize_symbol(symbol) if symbol else None
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
    lst = [normalize_symbol(s.strip()) for s in symbols.split(",") if s.strip()]
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
    from ...strategies import STRATEGIES

    extras = {"cross_arbitrage"}
    all_names = list(STRATEGIES.keys()) + list(extras)
    return {
        "strategies": {
            name: _strategies_state.get(name, "stopped") for name in all_names
        }
    }


@app.get("/strategies/info")
def strategies_info():
    """Return metadata for available strategies."""
    from ...strategies import STRATEGY_INFO

    return STRATEGY_INFO


@app.get("/strategies/{name}/schema")
def strategy_schema(name: str):
    """Return constructor parameters and defaults for a strategy.

    The endpoint inspects the strategy's ``__init__`` signature and returns
    parameter names, annotated types and default values.  Strategies that use
    ``**kwargs`` with in-class defaults are instantiated so their attributes can
    be discovered.
    """

    import importlib
    import inspect

    try:
        module = importlib.import_module(f"tradingbot.strategies.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail="unknown strategy") from exc

    strategy_cls = None
    for attr in dir(module):
        obj = getattr(module, attr)
        if inspect.isclass(obj) and getattr(obj, "name", None) == name:
            strategy_cls = obj
            break
    if strategy_cls is None:
        raise HTTPException(status_code=404, detail="strategy class not found")

    sig = inspect.signature(strategy_cls.__init__)
    params: list[dict] = []
    for p in list(sig.parameters.values())[1:]:  # skip ``self``
        if p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        default = None if p.default is inspect._empty else p.default
        if p.annotation is inspect._empty:
            annotation = "Any"
        elif isinstance(p.annotation, type):
            annotation = p.annotation.__name__
        else:
            annotation = str(p.annotation)
        params.append({"name": p.name, "type": annotation, "default": default})

    # Strategies relying solely on ``**kwargs`` won't expose parameters via the
    # signature.  Attempt to instantiate the class and inspect its attributes to
    # derive defaults in that case.
    if not params:
        try:  # pragma: no cover - best effort
            inst = strategy_cls()
            for k, v in vars(inst).items():
                if k.startswith("_"):
                    continue
                params.append({"name": k, "type": type(v).__name__, "default": v})
        except Exception:
            pass

    return {"strategy": name, "params": params}


# --- Funding and basis endpoints ------------------------------------------------


@app.get("/funding")
def get_funding():
    """Return latest funding rates by symbol."""

    return {"funding": _funding}


@app.post("/funding")
def set_funding(data: dict[str, float]):
    """Update funding rates for one or more symbols."""
    data = {normalize_symbol(k): v for k, v in data.items()}
    _funding.update(data)
    return {"funding": _funding}


@app.get("/basis")
def get_basis():
    """Return latest basis values by symbol."""

    return {"basis": _basis}


@app.post("/basis")
def set_basis(data: dict[str, float]):
    """Update basis values for one or more symbols."""
    data = {normalize_symbol(k): v for k, v in data.items()}
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
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=env)
        return {
            "command": req.command,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except Exception as exc:  # pragma: no cover - subprocess failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Background CLI processes keyed by an ID
_CLI_PROCS: dict[str, asyncio.subprocess.Process] = {}


@app.post("/cli/start")
async def cli_start(req: CLIRequest):
    """Start a CLI command in the background and return an ID."""

    args = shlex.split(req.command)
    cmd = [sys.executable, "-u", "-m", "tradingbot.cli", *args]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    job_id = uuid.uuid4().hex
    _CLI_PROCS[job_id] = proc
    return {"id": job_id}


async def _stream_process(proc: asyncio.subprocess.Process):
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def reader(stream: asyncio.StreamReader):
        while True:
            line = await stream.readline()
            if not line:
                break
            await queue.put(line.decode())

    tasks = [
        asyncio.create_task(reader(proc.stdout)),
        asyncio.create_task(reader(proc.stderr)),
    ]

    try:
        while True:
            if proc.stdout.at_eof() and proc.stderr.at_eof() and queue.empty():
                break
            try:
                line = await asyncio.wait_for(queue.get(), 0.1)
            except asyncio.TimeoutError:
                continue
            yield f"data: {line.rstrip()}\n\n"
    finally:
        for t in tasks:
            t.cancel()
        # Ensure the process is fully awaited so that ``proc.returncode``
        # is available for the caller of :func:`cli_stream`.
        await proc.wait()


@app.get("/cli/stream/{job_id}")
async def cli_stream(job_id: str):
    """Stream output from a running CLI job as server-sent events."""

    proc = _CLI_PROCS.get(job_id)
    if not proc:
        raise HTTPException(status_code=404, detail="job not found")

    async def event_gen():
        try:
            async for chunk in _stream_process(proc):
                yield chunk
        finally:
            # Remove job and always emit an ``end`` event so that the client
            # knows the stream is finished even if many lines were produced.
            _CLI_PROCS.pop(job_id, None)
            returncode = proc.returncode if proc.returncode is not None else 0
            yield f"event: end\ndata: {returncode}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/cli/stop/{job_id}")
async def cli_stop(job_id: str):
    """Terminate a running CLI job."""

    proc = _CLI_PROCS.get(job_id)
    if not proc:
        raise HTTPException(status_code=404, detail="job not found")
    proc.terminate()
    return {"status": "terminated"}


# --- Bot lifecycle --------------------------------------------------------------


class BotConfig(BaseModel):
    """Configuration for launching a trading bot."""

    strategy: str
    pairs: list[str] | None = None
    notional: float | None = None
    venue: str | None = None
    equity_pct: float | None = None
    leverage: int | None = None
    risk_pct: float | None = None
    daily_max_loss_pct: float | None = None
    daily_max_drawdown_pct: float | None = None
    testnet: bool | None = None
    dry_run: bool | None = None
    spot: str | None = None
    perp: str | None = None
    threshold: float | None = None


_BOTS: dict[int, dict] = {}


def _build_bot_args(cfg: BotConfig) -> list[str]:
    # Special runner for cross exchange arbitrage / cash and carry
    if cfg.strategy == "cross_arbitrage":
        if not cfg.pairs:
            raise ValueError("pairs required for cross_arbitrage")
        if not (cfg.spot and cfg.perp):
            raise ValueError("spot and perp adapters required")
        args = [
            sys.executable,
            "-m",
            "tradingbot.cli",
            "run-cross-arb",
            normalize_symbol(cfg.pairs[0]),
            cfg.spot,
            cfg.perp,
        ]
        if cfg.threshold is not None:
            args.extend(["--threshold", str(cfg.threshold)])
        if cfg.notional is not None:
            args.extend(["--notional", str(cfg.notional)])
        return args

    args = [
        sys.executable,
        "-m",
        "tradingbot.cli",
        "run-bot",
        "--strategy",
        cfg.strategy,
    ]
    for pair in cfg.pairs or []:
        args.extend(["--symbol", normalize_symbol(pair)])
    if cfg.notional is not None:
        args.extend(["--notional", str(cfg.notional)])
    if cfg.venue:
        args.extend(["--venue", cfg.venue])
    if cfg.equity_pct is not None:
        args.extend(["--equity-pct", str(cfg.equity_pct)])
    if cfg.leverage is not None:
        args.extend(["--leverage", str(cfg.leverage)])
    if cfg.risk_pct is not None:
        args.extend(["--risk-pct", str(cfg.risk_pct)])
    if cfg.daily_max_loss_pct is not None:
        args.extend(["--daily-max-loss-pct", str(cfg.daily_max_loss_pct)])
    if cfg.daily_max_drawdown_pct is not None:
        args.extend(["--daily-max-drawdown-pct", str(cfg.daily_max_drawdown_pct)])
    if cfg.testnet is not None:
        args.append("--testnet" if cfg.testnet else "--no-testnet")
    if cfg.dry_run is not None:
        args.append("--dry-run" if cfg.dry_run else "--no-dry-run")
    return args


@app.post("/bots")
async def start_bot(cfg: BotConfig):
    """Launch a bot process using the provided configuration."""

    args = _build_bot_args(cfg)
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _BOTS[proc.pid] = {"process": proc, "config": cfg.dict()}
    return {"pid": proc.pid, "status": "started"}


@app.get("/bots")
def list_bots(request: Request):
    """Return information about running bot processes."""

    if "text/html" in request.headers.get("accept", ""):
        try:
            html = (_static_dir / "bots.html").read_text(encoding="utf-8")
            return HTMLResponse(content=html)
        except Exception:
            return HTMLResponse(content="Bots dashboard not found", status_code=404)

    items = []
    for pid, info in _BOTS.items():
        proc = info["process"]
        status = "running" if proc.returncode is None else f"stopped:{proc.returncode}"
        items.append({"pid": pid, "status": status, "config": info["config"]})
    return {"bots": items}


@app.post("/bots/{pid}/stop")
async def stop_bot(pid: int):
    """Terminate a running bot process."""

    info = _BOTS.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    proc: asyncio.subprocess.Process = info["process"]
    if proc.returncode is not None:
        raise HTTPException(status_code=400, detail="bot not running")
    proc.terminate()
    await proc.wait()
    return {"pid": pid, "status": "stopped", "returncode": proc.returncode}


@app.post("/bots/{pid}/pause")
def pause_bot(pid: int):
    """Send SIGSTOP to a running bot process."""

    info = _BOTS.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    proc: asyncio.subprocess.Process = info["process"]
    if proc.returncode is not None:
        raise HTTPException(status_code=400, detail="bot not running")
    proc.send_signal(signal.SIGSTOP)
    return {"pid": pid, "status": "paused"}


@app.post("/bots/{pid}/resume")
def resume_bot(pid: int):
    """Send SIGCONT to resume a paused bot."""

    info = _BOTS.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    proc: asyncio.subprocess.Process = info["process"]
    if proc.returncode is not None:
        raise HTTPException(status_code=400, detail="bot not running")
    proc.send_signal(signal.SIGCONT)
    return {"pid": pid, "status": "running"}


@app.delete("/bots/{pid}")
async def delete_bot(pid: int):
    """Remove process information and terminate if still running."""

    info = _BOTS.pop(pid, None)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    proc: asyncio.subprocess.Process = info["process"]
    if proc.returncode is None:
        proc.terminate()
        await proc.wait()
    return {"pid": pid, "status": "deleted", "returncode": proc.returncode}

