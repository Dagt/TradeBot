import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus
from ..utils.metrics import FUNDING_RATE, FUNDING_RATE_HIST

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
    sane defaults so callers only need to handle the happy path.  ``adapter``
    may return either a plain mapping or one of the Pydantic models used by
    the exchange connectors.
    """

    fetch = getattr(adapter, "fetch_funding", None) or getattr(
        adapter, "fetch_funding_rate", None
    )
    if fetch is None:
        raise AttributeError("adapter lacks fetch_funding")

    info: Any = await fetch(symbol)
    if isinstance(info, dict):
        ts_raw = info.get("ts") or info.get("time") or info.get("timestamp")
        rate_raw = info.get("rate") or info.get("fundingRate") or info.get("value")
        interval_raw = info.get("interval_sec") or info.get("interval")
        exchange = info.get("exchange") or getattr(adapter, "name", "unknown")
    else:
        ts_raw = (
            getattr(info, "timestamp", None)
            or getattr(info, "ts", None)
            or getattr(info, "time", None)
        )
        rate_raw = (
            getattr(info, "rate", None)
            or getattr(info, "fundingRate", None)
            or getattr(info, "value", None)
        )
        interval_raw = getattr(info, "interval_sec", getattr(info, "interval", None))
        exchange = getattr(info, "exchange", getattr(adapter, "name", "unknown"))

    if isinstance(ts_raw, datetime):
        ts = ts_raw
    elif isinstance(ts_raw, (int, float)):
        if ts_raw > 1e12:
            ts_raw /= 1000
        ts = datetime.fromtimestamp(ts_raw, timezone.utc)
    else:
        ts = datetime.now(timezone.utc)

    rate = float(rate_raw or 0.0)
    interval_sec = int(interval_raw or 0)
    return {"ts": ts, "rate": rate, "interval_sec": interval_sec, "exchange": exchange}


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
            FUNDING_RATE.labels(symbol=symbol).set(info["rate"])
            FUNDING_RATE_HIST.labels(symbol=symbol).observe(info["rate"])
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
