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
