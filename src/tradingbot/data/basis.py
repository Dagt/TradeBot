import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from ..bus import EventBus

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_basis
    _CAN_PG = True
except Exception:  # pragma: no cover - optional dependency
    _CAN_PG = False

log = logging.getLogger(__name__)


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
            info: Any = await adapter.fetch_basis(symbol)
            ts = info.get("ts") or datetime.now(timezone.utc)
            basis = float(info.get("basis") or 0.0)
            event = {
                "ts": ts,
                "exchange": getattr(adapter, "name", "unknown"),
                "symbol": symbol,
                "basis": basis,
            }
            await bus.publish("basis", event)
            if engine is not None:
                try:
                    insert_basis(engine, ts=ts, exchange=event["exchange"], symbol=symbol, basis=basis)
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar basis: %s", e)
        except asyncio.CancelledError:  # allow task cancellation
            raise
        except Exception as e:  # pragma: no cover - logging only
            log.warning("poll_basis error: %s", e)
        await asyncio.sleep(interval)
