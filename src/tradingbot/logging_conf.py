import logging, sys
from .config import settings

try:
    import sentry_sdk
except Exception:  # pragma: no cover - sentry optional
    sentry_sdk = None

def setup_logging():
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if settings.sentry_dsn and sentry_sdk is not None:
        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.env)
