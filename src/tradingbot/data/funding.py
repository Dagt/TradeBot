import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_funding
    _CAN_PG = True
except Exception:  # pragma: no cover - optional dependency
    _CAN_PG = False

log = logging.getLogger(__name__)

async def poll_funding(adapter, symbol: str, bus: EventBus, interval: int = 60, persist_pg: bool = False) -> None:
    """Poll periodic funding rates and publish them on the event bus.

    The ``adapter`` is expected to expose an asynchronous ``fetch_funding``
    method returning a mapping with at least ``rate`` and ``interval_sec``
    fields.  Each retrieved funding is published to the ``"funding"`` topic of
    the provided :class:`EventBus` and optionally persisted into TimescaleDB
    using :func:`insert_funding`.
    """
    engine = get_engine() if (persist_pg and _CAN_PG) else None
    while True:
        try:
            info: Any = await adapter.fetch_funding(symbol)
            rate = float(info.get("rate") or info.get("fundingRate") or 0.0)
            interval_sec = int(info.get("interval_sec") or info.get("interval", 0))
            ts = info.get("ts") or datetime.now(timezone.utc)
            event = {
                "ts": ts,
                "exchange": getattr(adapter, "name", "unknown"),
                "symbol": symbol,
                "rate": rate,
                "interval_sec": interval_sec,
            }
            await bus.publish("funding", event)
            if engine is not None:
                try:
                    insert_funding(engine, ts=ts, exchange=event["exchange"], symbol=symbol, rate=rate, interval_sec=interval_sec)
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar funding: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_funding error: %s", e)
        await asyncio.sleep(interval)
