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
            fetch = getattr(adapter, "fetch_open_interest", None) or getattr(adapter, "fetch_oi", None)
            if fetch is None:
                raise AttributeError("adapter lacks fetch_open_interest")
            info: Any = await fetch(symbol)
            ts_raw = info.get("ts") or info.get("time")
            if isinstance(ts_raw, (int, float)):
                ts = datetime.fromtimestamp(ts_raw / 1000, timezone.utc)
            else:
                ts = ts_raw or datetime.now(timezone.utc)
            oi = float(info.get("oi") or info.get("openInterest") or 0.0)
            event = {
                "ts": ts,
                "exchange": getattr(adapter, "name", "unknown"),
                "symbol": symbol,
                "oi": oi,
            }
            await bus.publish("open_interest", event)
            if engine is not None:
                try:
                    insert_open_interest(engine, ts=ts, exchange=event["exchange"], symbol=symbol, oi=oi)
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar open interest: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_open_interest error: %s", e)
        await asyncio.sleep(interval)
