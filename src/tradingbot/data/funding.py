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

async def fetch_funding(adapter, symbol: str) -> dict[str, Any]:
    """Fetch a funding rate using ``adapter`` and normalise the result.

    The returned mapping includes ``ts`` (datetime), ``rate`` (float) and
    ``interval_sec`` (int) fields.  Any missing values are substituted with
    sane defaults so callers only need to handle the happy path.
    """

    info: Any = await adapter.fetch_funding(symbol)
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
    rate = float(
        data.get("rate")
            or data.get("fundingRate")
            or data.get("value")
            or 0.0
    )
    interval_sec = int(data.get("interval_sec") or data.get("interval", 0))
    return {
        "ts": ts,
        "rate": rate,
        "interval_sec": interval_sec,
        "exchange": getattr(adapter, "name", "unknown"),
    }


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
            info = await fetch_funding(adapter, symbol)
            event = {
                "ts": info["ts"],
                "exchange": info["exchange"],
                "symbol": symbol,
                "rate": info["rate"],
                "interval_sec": info["interval_sec"],
            }
            await bus.publish("funding", event)
            if engine is not None:
                try:
                    insert_funding(
                        engine,
                        ts=info["ts"],
                        exchange=event["exchange"],
                        symbol=symbol,
                        rate=info["rate"],
                        interval_sec=info["interval_sec"],
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar funding: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_funding error: %s", e)
        await asyncio.sleep(interval)
