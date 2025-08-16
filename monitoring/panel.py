from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any

from .metrics import router as metrics_router

# Optional Sentry integration
try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except Exception:  # pragma: no cover - sentry optional
    sentry_sdk = None
    FastApiIntegration = None


def _load_sentry_config(path: Path) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if not path.exists():
        return config
    raw = path.read_text().splitlines()
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        # basic type casting
        if value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        else:
            try:
                config[key] = float(value)
            except ValueError:
                config[key] = value
    return config


def _init_sentry():  # pragma: no cover - side effects
    if sentry_sdk is None:
        return
    config = _load_sentry_config(Path(__file__).with_name("sentry.yml"))
    if not config.get("dsn"):
        return
    sentry_sdk.init(
        dsn=config.get("dsn"),
        environment=config.get("environment"),
        traces_sample_rate=float(config.get("traces_sample_rate", 1.0)),
        integrations=[FastApiIntegration()] if FastApiIntegration else None,
    )


_init_sentry()

app = FastAPI(title="TradeBot Monitoring")
app.include_router(metrics_router)

static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
