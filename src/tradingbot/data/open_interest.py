import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus
from ..utils.metrics import OPEN_INTEREST, OPEN_INTEREST_HIST

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_open_interest
    _CAN_PG = True
except Exception:  # pragma: no cover - optional dependency
    _CAN_PG = False

log = logging.getLogger(__name__)


async def fetch_oi(adapter, symbol: str) -> dict[str, Any]:
    """Fetch open interest using ``adapter`` and return a normalised mapping."""

    fetch = getattr(adapter, "fetch_open_interest", None) or getattr(
        adapter, "fetch_oi", None
    )
    if fetch is None:
        raise AttributeError("adapter lacks fetch_open_interest")

    info: Any = await fetch(symbol)
    if isinstance(info, dict):
        ts_raw = info.get("ts") or info.get("time") or info.get("timestamp")
        oi_raw = info.get("oi") or info.get("openInterest") or info.get("value")
        exchange = info.get("exchange") or getattr(adapter, "name", "unknown")
    else:
        ts_raw = (
            getattr(info, "timestamp", None)
            or getattr(info, "ts", None)
            or getattr(info, "time", None)
        )
        oi_raw = (
            getattr(info, "oi", None)
            or getattr(info, "openInterest", None)
            or getattr(info, "value", None)
        )
        exchange = getattr(info, "exchange", getattr(adapter, "name", "unknown"))

    if isinstance(ts_raw, datetime):
        ts = ts_raw
    elif isinstance(ts_raw, (int, float)):
        if ts_raw > 1e12:
            ts_raw /= 1000
        ts = datetime.fromtimestamp(ts_raw, timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    oi = float(oi_raw or 0.0)
    return {"ts": ts, "oi": oi, "exchange": exchange}


async def poll_open_interest(
    adapter,
    symbol: str,
    bus: EventBus,
    interval: int = 60,
    persist_pg: bool = False,
) -> None:
    """Poll periodic open interest and publish on the event bus."""
    engine = get_engine() if (persist_pg and _CAN_PG) else None
    while True:
        try:
            info = await fetch_oi(adapter, symbol)
            event = {
                "ts": info["ts"],
                "exchange": info["exchange"],
                "symbol": symbol,
                "oi": info["oi"],
            }
            OPEN_INTEREST.labels(symbol=symbol).set(info["oi"])
            OPEN_INTEREST_HIST.labels(symbol=symbol).observe(info["oi"])
            await bus.publish("open_interest", event)
            if engine is not None:
                try:
                    insert_open_interest(
                        engine,
                        ts=info["ts"],
                        exchange=event["exchange"],
                        symbol=symbol,
                        oi=info["oi"],
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar open interest: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_open_interest error: %s", e)
        await asyncio.sleep(interval)
