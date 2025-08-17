import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus
from ..utils.metrics import BASIS, BASIS_HIST

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_basis
    _CAN_PG = True
except Exception:  # pragma: no cover - optional dependency
    _CAN_PG = False

log = logging.getLogger(__name__)


async def fetch_basis(adapter, symbol: str) -> dict[str, Any]:
    """Fetch basis information using ``adapter`` and normalise the output."""

    fetch = getattr(adapter, "fetch_basis", None)
    if fetch is None:
        raise AttributeError("adapter lacks fetch_basis")

    info: Any = await fetch(symbol)
    ts_raw = info.get("ts") or info.get("time")
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(ts_raw / 1000, tz=timezone.utc)
    else:
        ts = ts_raw or datetime.now(timezone.utc)
    basis = float(info.get("basis") or info.get("value") or 0.0)
    return {
        "ts": ts,
        "basis": basis,
        "exchange": getattr(adapter, "name", "unknown"),
    }


async def poll_basis(
    adapter,
    symbol: str,
    bus: EventBus,
    interval: int = 60,
    persist_pg: bool = False,
) -> None:
    """Poll periodic basis values and publish on the event bus."""
    engine = get_engine() if (persist_pg and _CAN_PG) else None
    while True:
        try:
            info = await fetch_basis(adapter, symbol)
            event = {
                "ts": info["ts"],
                "exchange": info["exchange"],
                "symbol": symbol,
                "basis": info["basis"],
            }
            BASIS.labels(symbol=symbol).set(info["basis"])
            BASIS_HIST.labels(symbol=symbol).observe(info["basis"])
            await bus.publish("basis", event)
            if engine is not None:
                try:
                    insert_basis(
                        engine,
                        ts=info["ts"],
                        exchange=event["exchange"],
                        symbol=symbol,
                        basis=info["basis"],
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar basis: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_basis error: %s", e)
        await asyncio.sleep(interval)
