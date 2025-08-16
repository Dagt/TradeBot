from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .metrics import router as metrics_router
from .strategies import router as strategies_router

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)
app.include_router(strategies_router)

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
