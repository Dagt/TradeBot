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
import logging
import json
import re
from time import monotonic
from contextlib import suppress
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from collections import deque

try:  # pragma: no cover - ccxt is optional
    import ccxt  # type: ignore
    from ccxt.base.errors import AuthenticationError  # type: ignore
except Exception:  # pragma: no cover - if ccxt is missing
    ccxt = None
    class AuthenticationError(Exception):
        pass

from monitoring.metrics import (
    router as metrics_router,
    CLI_PROCESS_COMPLETED,
    CLI_PROCESS_TIMEOUT,
)
from monitoring.dashboard import router as dashboard_router

from ...storage.timescale import select_recent_fills
from ...utils.metrics import REQUEST_COUNT, REQUEST_LATENCY
from ...config import settings
from ...cli.utils import get_adapter_class, get_supported_kinds
from ...exchanges import SUPPORTED_EXCHANGES
from ...core.symbols import normalize as normalize_symbol
from ...connectors import (
    BinanceConnector,
    BinanceFuturesConnector,
    OKXConnector,
    BybitConnector,
)
from ...utils import secrets as secrets_utils


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
logger = logging.getLogger(__name__)

CONNECTORS = {
    "binance": BinanceConnector,
    "binance_testnet": BinanceConnector,
    "okx": OKXConnector,
    "okx_testnet": OKXConnector,
    "bybit": BybitConnector,
    "bybit_testnet": BybitConnector,
    "binance_futures": BinanceFuturesConnector,
    "binance_futures_testnet": BinanceFuturesConnector,
}


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


@app.get("/balances/{exchange}")
async def get_balances(exchange: str):
    """Return account balances for one or multiple exchanges."""
    if ccxt is None:
        raise HTTPException(status_code=503, detail="ccxt not available")
    exchanges = [e.strip() for e in exchange.split(",") if e.strip()]
    results: dict[str, dict] = {}
    for ex in exchanges:
        key = ex.lower()
        cls = CONNECTORS.get(key)
        if cls is None:
            raise HTTPException(status_code=404, detail=f"unsupported exchange: {ex}")
        conn = cls()
        testnet = key.endswith("_testnet")
        base_key = key[:-8] if testnet else key
        api_key, api_secret = secrets_utils.get_api_credentials(base_key)
        if not api_key or not api_secret:
            logger.warning("missing_credentials", extra={"exchange": ex})
            results[ex] = {"error": "missing_credentials"}
            with suppress(Exception):
                await conn.rest.close()
            if len(exchanges) == 1:
                raise HTTPException(status_code=403, detail="missing credentials")
            continue
        conn.rest.apiKey = api_key
        conn.rest.secret = api_secret
        if testnet:
            with suppress(Exception):
                conn.rest.set_sandbox_mode(True)
        try:
            results[ex] = await conn.fetch_balance()
        except AuthenticationError as e:  # pragma: no cover - auth issues
            logger.warning("balance_auth_error", extra={"exchange": ex, "err": str(e)})
            results[ex] = {"error": str(e)}
            if len(exchanges) == 1:
                raise HTTPException(status_code=403, detail="authentication failed")
        except Exception as e:  # pragma: no cover - network issues
            logger.warning("balance_error", extra={"exchange": ex, "err": str(e)})
            results[ex] = {"error": str(e)}
        finally:
            with suppress(Exception):
                await conn.rest.close()
    if len(results) == 1 and exchange == exchanges[0]:
        return results[exchanges[0]]
    return results


@app.get("/tickers/{exchange}")
async def get_tickers(exchange: str, symbols: str = Query(...)):
    """Return ticker price for each requested symbol."""
    if ccxt is None:
        raise HTTPException(status_code=503, detail="ccxt not available")
    key = exchange.lower()
    cls = CONNECTORS.get(key)
    if cls is None:
        raise HTTPException(status_code=404, detail="unsupported exchange")
    conn = cls()
    testnet = key.endswith("_testnet")
    if testnet:
        with suppress(Exception):
            conn.rest.set_sandbox_mode(True)
    prices: dict[str, float | None] = {}
    # Ensure markets are loaded so that symbols can be mapped to the
    # exchange-specific format (e.g. ``BTC/USDT:USDT`` on Bybit).
    await conn.rest.load_markets()  # pragma: no cover - network issues
    for raw in symbols.split(","):
        sym = raw.strip()
        if not sym:
            continue
        norm = normalize_symbol(sym)
        try:
            mapped = conn.rest.market_id(norm)
        except Exception:
            mapped = norm
        try:
            if hasattr(conn, "fetch_ticker"):
                price = await conn.fetch_ticker(mapped)  # type: ignore[func-returns-value]
            else:
                data = await conn.rest.fetch_ticker(mapped)
                price = (
                    data.get("last")
                    or data.get("close")
                    or data.get("price")
                    or data.get("bid")
                    or data.get("ask")
                )
            prices[norm] = float(price) if price is not None else None
        except Exception as e:  # pragma: no cover - network issues
            logger.warning(
                "ticker_error",
                extra={"exchange": exchange, "symbol": sym, "err": str(e)},
            )
            prices[norm] = None
    with suppress(Exception):
        await conn.rest.close()
    return prices


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


class SecretItem(BaseModel):
    value: str


@app.get("/secrets/{key}")
def get_secret(key: str):
    val = secrets_utils.read_secret(key)
    if val is None:
        raise HTTPException(status_code=404, detail="secret not found")
    return {"value": val}


@app.post("/secrets/{key}")
def set_secret(key: str, payload: SecretItem):
    secrets_utils.set_secret(key, payload.value)
    return {"status": "ok"}


@app.delete("/secrets/{key}")
def delete_secret(key: str):
    secrets_utils.delete_secret(key)
    return {"status": "deleted"}


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
        rm = getattr(app.state, "risk_service", None)
        bus = getattr(rm, "bus", None) if rm else None
    if bus is None:
        raise HTTPException(status_code=500, detail="bus not configured")
    await bus.publish("control:halt", {"reason": payload.reason})
    return {"status": "halt"}


@app.post("/risk/reset")
def risk_reset():
    rm = getattr(app.state, "risk_service", None)
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

    import dataclasses
    import importlib
    import inspect
    import re

    from ...strategies import STRATEGIES

    # Parameters that should never be exposed via the schema endpoint
    _INTERNAL_KEYS = {
        "config_path",
        "risk_service",
        "spot_exchange",
        "perp_exchange",
        "persist_pg",
        "rebalance_assets",
        "rebalance_threshold",
        "latency",
    }

    # Special case for the functional cross exchange arbitrage strategy
    if name == "cross_arbitrage":
        from ...strategies.cross_exchange_arbitrage import (
            CrossArbConfig,
            PARAM_INFO as param_info,
        )

        param_info = {k: v for k, v in param_info.items() if k not in _INTERNAL_KEYS}

        params: list[dict] = []
        for f in dataclasses.fields(CrossArbConfig):
            if f.name in _INTERNAL_KEYS:
                continue
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                default = None
            else:
                default = (
                    f.default
                    if f.default is not dataclasses.MISSING
                    else f.default_factory()
                )
            annotation = f.type.__name__ if isinstance(f.type, type) else str(f.type)
            desc = param_info.get(f.name, "")
            params.append(
                {"name": f.name, "type": annotation, "default": default, "desc": desc}
            )
        return {"strategy": name, "params": params}

    strategy_cls = STRATEGIES.get(name)
    module = None
    if strategy_cls is not None:
        module = importlib.import_module(strategy_cls.__module__)
    else:
        try:
            module = importlib.import_module(f"tradingbot.strategies.{name}")
        except ModuleNotFoundError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail="unknown strategy") from exc
        for attr in dir(module):
            obj = getattr(module, attr)
            if inspect.isclass(obj) and getattr(obj, "name", None) == name:
                strategy_cls = obj
                break
    if strategy_cls is None:
        raise HTTPException(status_code=404, detail="strategy class not found")

    sig = inspect.signature(strategy_cls.__init__)
    params: list[dict] = []

    # Collect optional descriptions from PARAM_INFO and docstrings
    param_info: dict[str, str] = {}
    for source in (
        getattr(module, "PARAM_INFO", None),
        getattr(strategy_cls, "PARAM_INFO", None),
    ):
        if isinstance(source, dict):
            param_info.update({str(k): str(v) for k, v in source.items()})

    # Drop internal keys from optional descriptions
    for k in list(param_info):
        if k in _INTERNAL_KEYS:
            del param_info[k]

    doc_params: dict[str, str] = {}
    doc = inspect.getdoc(strategy_cls.__init__) or inspect.getdoc(strategy_cls)
    if doc:
        for m in re.finditer(r":param\s+(\w+)\s*:\s*(.+)", doc):
            if m.group(1) not in _INTERNAL_KEYS:
                doc_params[m.group(1)] = m.group(2).strip()

    for p in list(sig.parameters.values())[1:]:  # skip ``self``
        if p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        if p.name in _INTERNAL_KEYS:
            continue
        default = None if p.default is inspect._empty else p.default
        if p.annotation is inspect._empty:
            annotation = "Any"
        elif isinstance(p.annotation, type):
            annotation = p.annotation.__name__
        else:
            annotation = str(p.annotation)
        desc = param_info.get(p.name) or doc_params.get(p.name) or ""
        params.append(
            {"name": p.name, "type": annotation, "default": default, "desc": desc}
        )
    # Expand dataclass configs if only ``cfg`` was captured
    if params and len(params) == 1 and params[0]["name"] == "cfg":
        try:  # pragma: no cover - best effort
            import dataclasses

            inst = strategy_cls()
            cfg = getattr(inst, "cfg", None)
            if dataclasses.is_dataclass(cfg):
                params = []
                for f in dataclasses.fields(cfg):
                    field_name = f.name
                    if field_name in _INTERNAL_KEYS or field_name not in param_info:
                        continue
                    default = getattr(cfg, field_name)
                    desc = param_info.get(field_name, "")
                    params.append(
                        {
                            "name": field_name,
                            "type": type(default).__name__,
                            "default": default,
                            "desc": desc,
                        }
                    )
        except Exception:
            pass
    # Strategies relying solely on ``**kwargs`` won't expose parameters via the
    # signature.  Attempt to instantiate the class and inspect its attributes to
    # derive defaults in that case.
    if not params:
        try:  # pragma: no cover - best effort
            import dataclasses

            inst = strategy_cls()
            for attr_name, v in vars(inst).items():
                if attr_name.startswith("_") or attr_name in _INTERNAL_KEYS:
                    continue
                if dataclasses.is_dataclass(v):
                    for f in dataclasses.fields(v):
                        field_name = f.name
                        if field_name in _INTERNAL_KEYS:
                            continue
                        default = getattr(v, field_name)
                        if field_name not in param_info and field_name not in doc_params:
                            continue
                        desc = param_info.get(field_name) or doc_params.get(field_name) or ""
                        params.append(
                            {
                                "name": field_name,
                                "type": type(default).__name__,
                                "default": default,
                                "desc": desc,
                            }
                        )
                    continue
                if attr_name not in param_info and attr_name not in doc_params:
                    continue
                desc = param_info.get(attr_name) or doc_params.get(attr_name) or ""
                params.append(
                    {
                        "name": attr_name,
                        "type": type(v).__name__,
                        "default": v,
                        "desc": desc,
                    }
                )
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

# By default, CLI commands run without a timeout. Clients may specify one if desired.
DEFAULT_CLI_TIMEOUT: int | None = None

class CLIRequest(BaseModel):
    """Payload for running a command via the CLI.

    Parameters
    ----------
    command:
        CLI arguments to pass to ``python -m tradingbot.cli``.
    timeout:
        Optional limit in seconds for command execution. If omitted, no
        timeout is enforced.
    """

    command: str
    timeout: int | None = Field(
        default=None,
        description="Maximum seconds to allow the command to run. Unlimited if omitted.",
        ge=1,
    )


@app.post("/cli/run")
def run_cli(req: CLIRequest):
    """Execute a TradingBot CLI command and return its output.

    The endpoint runs ``python -m tradingbot.cli`` with the arguments supplied
    in ``command``.  Output from ``stdout`` and ``stderr`` is captured and
    returned to the caller so it can be displayed in the dashboard.

    Clients may set ``timeout`` in the request body to limit the execution
    time. If omitted, the command runs without a timeout.
    """

    args = shlex.split(req.command)
    cmd = [sys.executable, "-m", "tradingbot.cli", *args]
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
    timeout = req.timeout if req.timeout is not None else DEFAULT_CLI_TIMEOUT
    run_kwargs = dict(capture_output=True, text=True, env=env)
    if timeout is not None:
        run_kwargs["timeout"] = timeout
    try:
        res = subprocess.run(cmd, **run_kwargs)
        return {
            "command": req.command,
            "returncode": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except Exception as exc:  # pragma: no cover - subprocess failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Background CLI jobs keyed by an ID
# Each job stores the process, optional timeout and start time for enforcement.
_CLI_JOBS: dict[str, dict[str, object]] = {}


@app.post("/cli/start")
async def cli_start(req: CLIRequest):
    """Start a CLI command in the background and return an ID.

    The optional ``timeout`` in the request is stored with the job so that it
    can be enforced during streaming.
    """

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
    _CLI_JOBS[job_id] = {
        "proc": proc,
        "timeout": req.timeout,
        "start": time.perf_counter(),
    }
    return {"id": job_id}


def format_sse(event: str, payload: str) -> str:
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_process(
    proc: asyncio.subprocess.Process,
    job_id: str,
    timeout: int | None,
    start: float,
):
    queue: asyncio.Queue[str] = asyncio.Queue()
    running = True
    ended = False

    async def reader(stream: asyncio.StreamReader):
        nonlocal running
        while running:
            line = await stream.readline()
            if not line:
                break
            await queue.put(line.decode())

    tasks = [
        asyncio.create_task(reader(proc.stdout)),
        asyncio.create_task(reader(proc.stderr)),
    ]

    HEARTBEAT_SECS = 5
    last_emit = monotonic()

    try:
        while running:
            if timeout is not None and time.perf_counter() - start > timeout and proc.returncode is None:
                proc.terminate()

            try:
                line = queue.get_nowait()
            except asyncio.QueueEmpty:
                line = None

            if line is not None:
                line = line.rstrip()
                yield format_sse("message", line)
                last_emit = monotonic()
                if "Backtest finalizado" in line:
                    yield format_sse("status", "done")
                    yield format_sse("end", "")
                    ended = True
                    running = False
                    queue.put_nowait("")
                    break
                continue

            if proc.returncode is not None:
                with suppress(Exception):
                    await proc.wait()
                while True:
                    try:
                        line = queue.get_nowait()
                        yield format_sse("message", line.rstrip())
                    except asyncio.QueueEmpty:
                        break
                status_payload = json.dumps(
                    {"job_id": job_id, "status": "finished", "returncode": proc.returncode}
                )
                yield format_sse("status", status_payload)
                running = False
                queue.put_nowait("")
                break

            now = monotonic()
            if now - last_emit >= HEARTBEAT_SECS:
                yield format_sse("heartbeat", "")
                last_emit = now

            await asyncio.sleep(0.1)
    except Exception as exc:  # pragma: no cover - defensive
        yield format_sse("error", str(exc))
        running = False
    finally:
        running = False
        for t in tasks:
            t.cancel()
        with suppress(Exception):
            await proc.wait()
        if not ended:
            # ensure the client always receives a termination signal
            rc = proc.returncode
            yield format_sse("end", "" if rc is None else str(rc))


@app.get("/cli/stream/{job_id}")
async def cli_stream(job_id: str):
    """Stream output from a running CLI job as server-sent events."""

    job = _CLI_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    proc = job["proc"]
    timeout = job.get("timeout")
    start = job.get("start", time.perf_counter())

    async def event_gen():
        try:
            try:
                async for chunk in _stream_process(proc, job_id, timeout, start):
                    yield chunk
            except Exception as exc:
                yield format_sse("error", str(exc))
        finally:
            _CLI_JOBS.pop(job_id, None)
            await proc.wait()
            if proc.returncode == 0:
                CLI_PROCESS_COMPLETED.inc()
            else:
                CLI_PROCESS_TIMEOUT.inc()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/cli/stop/{job_id}")
async def cli_stop(job_id: str):
    """Terminate a running CLI job."""

    job = _CLI_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    job["proc"].terminate()
    return {"status": "terminated"}


@app.get("/cli/status/{job_id}")
async def cli_status(job_id: str):
    """Return the status of a CLI job."""

    job = _CLI_JOBS.get(job_id)
    if not job:
        return {"status": "not-found"}
    proc = job["proc"]
    if proc.returncode is None:
        return {"status": "running"}
    return {"status": "finished", "returncode": proc.returncode}


# --- Bot lifecycle --------------------------------------------------------------


class BotConfig(BaseModel):
    """Configuration for launching a trading bot.

    Strategies size their orders through a ``strength`` value, so explicit
    ``notional`` budgets are no longer required.
    """

    strategy: str
    pairs: list[str] | None = None
    venue: str | None = None
    leverage: int | None = None
    risk_pct: float | None = None
    daily_max_loss_pct: float | None = None
    daily_max_drawdown_pct: float | None = None
    testnet: bool | None = None
    dry_run: bool | None = None
    spot: str | None = None
    perp: str | None = None
    threshold: float | None = None
    config: str | None = None
    timeframe: str | None = None
    initial_cash: float | None = None
    metrics_port: int | None = None


_BOTS: dict[int, dict] = {}
_launch_lock = asyncio.Lock()

# Seconds to retain finished bot info and logs before purging.
# Allows a short window for clients to fetch final logs or status.
# Set ``BOT_LOG_RETENTION`` environment variable to override.
BOT_LOG_RETENTION: float = float(os.getenv("BOT_LOG_RETENTION", "1"))

# Regex para detectar líneas de métricas en logs: "METRICS {json}"
_METRIC_LINE = re.compile(r"METRICS?\s+(\{.*\})")


def update_bot_stats(pid: int, stats: dict | None = None, **kwargs) -> None:
    """Actualizar el buffer de métricas del bot ``pid``.

    Permite recibir métricas vía IPC o ser reutilizada por el scrapper de logs.
    ``stats`` puede ser un dict completo y ``kwargs`` métricas individuales.
    """

    info = _BOTS.get(pid)
    if not info:
        return
    buf = info.setdefault("stats", {})
    if stats:
        buf.update(stats)
    if kwargs:
        buf.update(kwargs)


async def _scrape_metrics(
    pid: int,
    proc: asyncio.subprocess.Process,
    log_buffer: deque[str] | None = None,
) -> None:
    """Lee stdout del proceso buscando líneas de métricas en JSON.

    Además, almacena todas las líneas en los buffers de logs si se proporcionan.
    """

    if not proc.stdout:
        return
    while True:
        try:
            line = await proc.stdout.readline()
        except Exception:
            break
        if not line:
            break
        text = line.decode(errors="ignore").rstrip()
        if log_buffer is not None:
            log_buffer.append(text)
        info = _BOTS.get(pid)
        queue = info.get("log_queue") if info else None
        if queue is not None:
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                pass
        m = _METRIC_LINE.search(text)
        if not m:
            continue
        try:
            payload = json.loads(m.group(1))
        except Exception:
            continue
        update_bot_stats(pid, payload)


async def _capture_stderr(
    pid: int,
    proc: asyncio.subprocess.Process,
    log_buffer: deque[str],
) -> None:
    if not proc.stderr:
        return
    while True:
        try:
            line = await proc.stderr.readline()
        except Exception:
            break
        if not line:
            break
        text = line.decode(errors="ignore").rstrip()
        log_buffer.append(text)
        info = _BOTS.get(pid)
        queue = info.get("log_queue") if info else None
        if queue is not None:
            try:
                queue.put_nowait(text)
            except asyncio.QueueFull:
                pass


async def _monitor_bot(pid: int, proc: asyncio.subprocess.Process) -> None:
    """Monitor a bot process and update its status when it finishes.

    The bot is removed from ``_BOTS`` after ``BOT_LOG_RETENTION`` seconds so
    clients can still fetch logs for a short while after termination.
    """

    try:
        await proc.wait()
    finally:
        info = _BOTS.get(pid)
        if not info:
            return
        # Preserve existing status if stop/kill updated it already
        if info.get("status") == "running":
            if proc.returncode == 0:
                info["status"] = "stopped"
            else:
                info["status"] = "error"
                info["last_error"] = f"return code {proc.returncode}"
        info["returncode"] = proc.returncode
        # Cancel log scraping tasks
        for name in ("metrics_task", "stderr_task"):
            task = info.get(name)
            if task:
                task.cancel()
                with suppress(Exception):
                    await task
        if BOT_LOG_RETENTION > 0:
            await asyncio.sleep(BOT_LOG_RETENTION)
        _BOTS.pop(pid, None)


def _build_bot_args(cfg: BotConfig, params: dict | None = None) -> list[str]:
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
        # El tamaño se deriva del diferencial spot/perp; no requiere --notional
        return args

    if cfg.dry_run and not cfg.testnet:
        if not cfg.pairs:
            raise ValueError("pairs required for paper trading")
        args = [
            sys.executable,
            "-m",
            "tradingbot.cli",
            "paper-run",
            "--symbol",
            normalize_symbol(cfg.pairs[0]),
            "--strategy",
            cfg.strategy,
        ]
        if cfg.venue:
            args.extend(["--venue", cfg.venue])
        if cfg.risk_pct is not None:
            args.extend(["--risk-pct", str(cfg.risk_pct)])
        if cfg.timeframe is not None:
            args.extend(["--timeframe", cfg.timeframe])
        if cfg.config:
            args.extend(["--config", cfg.config])
        if params:
            for k, v in params.items():
                args.extend(["--param", f"{k}={v}"])
        if cfg.initial_cash is not None:
            args.extend(["--initial-cash", str(cfg.initial_cash)])
        if cfg.metrics_port is not None:
            args.extend(["--metrics-port", str(cfg.metrics_port)])
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
    if cfg.venue:
        args.extend(["--venue", cfg.venue])
    if cfg.leverage is not None:
        args.extend(["--leverage", str(cfg.leverage)])
    if cfg.risk_pct is not None:
        args.extend(["--risk-pct", str(cfg.risk_pct)])
    if cfg.daily_max_loss_pct is not None:
        args.extend(["--daily-max-loss-pct", str(cfg.daily_max_loss_pct)])
    if cfg.daily_max_drawdown_pct is not None:
        args.extend(["--daily-max-drawdown-pct", str(cfg.daily_max_drawdown_pct)])
    if cfg.timeframe is not None:
        args.extend(["--timeframe", cfg.timeframe])
    if cfg.testnet is not None:
        args.append("--testnet" if cfg.testnet else "--no-testnet")
    if cfg.dry_run is not None:
        args.append("--dry-run" if cfg.dry_run else "--no-dry-run")
    if cfg.config:
        args.extend(["--config", cfg.config])
    if params:
        for k, v in params.items():
            args.extend(["--param", f"{k}={v}"])
    if cfg.metrics_port is not None:
        args.extend(["--metrics-port", str(cfg.metrics_port)])
    return args


@app.post("/bots")
async def start_bot(cfg: BotConfig, request: Request = None):
    """Launch a bot process using the provided configuration.

    For backward compatibility, derive ``venue`` from legacy ``exchange`` and
    ``market`` fields if provided in the request body.
    """

    async with _launch_lock:
        if cfg.venue is None and request is not None:
            try:
                data = await request.json()
            except Exception:  # pragma: no cover - malformed JSON
                data = {}
            exchange = data.get("exchange")
            market = data.get("market")
            if exchange and market:
                cfg.venue = f"{exchange}_{market}"

        cfg.metrics_port = cfg.metrics_port or 0
        params = _strategy_params.get(cfg.strategy, {})
        args = _build_bot_args(cfg, params)
        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = f"{repo_root}{os.pathsep}" + env.get("PYTHONPATH", "")
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        log_buffer: deque[str] = deque(maxlen=1000)
        stdout_task = asyncio.create_task(
            _scrape_metrics(proc.pid, proc, log_buffer)
        )
        stderr_task = asyncio.create_task(
            _capture_stderr(proc.pid, proc, log_buffer)
        )
        monitor_task = asyncio.create_task(
            _monitor_bot(proc.pid, proc)
        )
        _BOTS[proc.pid] = {
            "process": proc,
            "config": cfg.dict(),
            "stats": {},
            "status": "running",
            "metrics_task": stdout_task,
            "stderr_task": stderr_task,
             "monitor_task": monitor_task,
            "log_buffer": log_buffer,
        }
        return {"pid": proc.pid, "status": "running"}


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
        cfg = info["config"]
        stats = info.get("stats", {})
        risk_pct = cfg.get("risk_pct")
        if risk_pct is None:
            risk_pct = stats.get("risk_pct")
        items.append(
            {
                "pid": pid,
                "status": info.get("status"),
                "config": cfg,
                "stats": stats,
                "risk_pct": risk_pct,
                "last_error": info.get("last_error"),
            }
        )
    return {"bots": items}


@app.get("/bots/{pid}/logs")
async def bot_logs(pid: int):
    """Stream real-time logs from a bot using Server-Sent Events."""

    info = _BOTS.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    buffer: deque[str] | None = info.get("log_buffer")
    queue: asyncio.Queue[str] | None = info.get("log_queue")
    proc: asyncio.subprocess.Process = info["process"]
    if buffer is None:
        raise HTTPException(status_code=404, detail="logs not available")
    if queue is None:
        queue = asyncio.Queue(maxsize=1000)
        info["log_queue"] = queue

    async def event_gen():
        try:
            for line in list(buffer):
                yield format_sse("message", line)
            while True:
                if proc.returncode is not None and queue.empty():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                yield format_sse("message", line)
        finally:
            info["log_queue"] = None

    return StreamingResponse(event_gen(), media_type="text/event-stream")


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
    info["status"] = "stopped"
    return {"pid": pid, "status": info["status"], "returncode": proc.returncode}


@app.post("/bots/{pid}/kill")
async def kill_bot(pid: int):
    """Force kill a running bot process."""

    info = _BOTS.get(pid)
    if not info:
        raise HTTPException(status_code=404, detail="bot not found")
    proc: asyncio.subprocess.Process = info["process"]
    if proc.returncode is not None:
        raise HTTPException(status_code=400, detail="bot not running")
    proc.kill()
    await proc.wait()
    info["status"] = "killed"
    return {"pid": pid, "status": info["status"], "returncode": proc.returncode}


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
    info["status"] = "paused"
    return {"pid": pid, "status": info["status"]}


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
    info["status"] = "running"
    return {"pid": pid, "status": info["status"]}


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
    task = info.get("metrics_task")
    if task:
        task.cancel()
    err_task = info.get("stderr_task")
    if err_task:
        err_task.cancel()
    mon_task = info.get("monitor_task")
    if mon_task:
        mon_task.cancel()
    return {"pid": pid, "status": "deleted", "returncode": proc.returncode}

