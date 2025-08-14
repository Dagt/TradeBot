# src/tradingbot/apps/api/main.py
from fastapi import FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Persistencia
try:
    from ...storage.timescale import get_engine, select_recent_tri_signals, select_recent_orders
    _CAN_PG = True
    _ENGINE = get_engine()
except Exception:
    _CAN_PG = False
    _ENGINE = None

app = FastAPI(title="TradingBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "db": bool(_CAN_PG)}

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

@app.get("/pnl/summary")
def pnl_summary(venue: str = "binance_spot_testnet"):
    if not _CAN_PG:
        return {"venue": venue, "items": [], "totals": {"upnl":0,"rpnl":0,"fees":0,"net_pnl":0}}
    from ...storage.timescale import select_pnl_summary
    return select_pnl_summary(_ENGINE, venue=venue)
