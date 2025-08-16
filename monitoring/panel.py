from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import yaml
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

from .metrics import router as metrics_router
from .strategies import router as strategies_router

config_path = Path(__file__).with_name("sentry.yml")
if config_path.exists():
    with config_path.open() as f:
        config = yaml.safe_load(f)
    sentry_sdk.init(integrations=[FastApiIntegration()], **config)

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)
app.include_router(strategies_router)

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
