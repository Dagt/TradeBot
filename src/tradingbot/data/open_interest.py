import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus

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
    if hasattr(info, "model_dump"):
        data = info.model_dump()
    elif hasattr(info, "dict"):
        data = info.dict()
    elif isinstance(info, dict):
        data = info
    else:
        data = {}
    ts_raw = data.get("ts") or data.get("timestamp") or data.get("time")
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(
            ts_raw / 1000 if ts_raw > 1e12 else ts_raw, timezone.utc
        )
    elif ts_raw is not None:
        ts = ts_raw
    else:
        ts = datetime.now(timezone.utc)
    oi = float(
        data.get("oi")
        or data.get("openInterest")
        or data.get("value")
        or 0.0
    )
    return {
        "ts": ts,
        "oi": oi,
        "exchange": getattr(adapter, "name", "unknown"),
    }


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
