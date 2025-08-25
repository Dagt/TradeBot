import logging, sys
from logging.handlers import RotatingFileHandler
from .config import settings

try:
    import sentry_sdk
except Exception:  # pragma: no cover - sentry optional
    sentry_sdk = None

try:  # pragma: no cover - optional dependency
    from pythonjsonlogger import jsonlogger
except Exception:
    jsonlogger = None


def setup_logging():
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if settings.log_file:
        handlers.append(
            RotatingFileHandler(
                settings.log_file,
                maxBytes=settings.log_max_bytes,
                backupCount=settings.log_backup_count,
            )
        )

    logging.basicConfig(level=level, handlers=handlers)

    if settings.log_json and jsonlogger is not None:
        formatter: logging.Formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    if settings.sentry_dsn and sentry_sdk is not None:
        sentry_sdk.init(dsn=settings.sentry_dsn, environment=settings.env)
