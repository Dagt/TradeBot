"""Ingestion workers that batch writes to storage with retries and metrics."""

from __future__ import annotations

import asyncio
import logging

from prometheus_client import Counter

log = logging.getLogger(__name__)

rows_inserted = Counter(
    "ingestion_rows_inserted", "Rows successfully persisted", ["table"]
)
failed_batches = Counter(
    "ingestion_failed_batches", "Batches that failed permanently", ["table"]
)


class BatchIngestionWorker:
    """Collects rows and writes them to storage in batches."""

    def __init__(
        self,
        client,
        table: str,
        insert_sql: str,
        batch_size: int = 100,
        max_retries: int = 3,
    ) -> None:
        self.client = client
        self.table = table
        self.insert_sql = insert_sql
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._buffer: list[dict] = []

    async def add(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            await self.flush()

    async def flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer
        self._buffer = []
        for attempt in range(1, self.max_retries + 1):
            try:
                await self.client.executemany(self.insert_sql, batch)
                rows_inserted.labels(self.table).inc(len(batch))
                return
            except Exception as exc:  # pragma: no cover - logging
                log.warning("insert failed attempt %s: %s", attempt, exc)
                if attempt >= self.max_retries:
                    failed_batches.labels(self.table).inc()
                    raise
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))


def _get_storage(backend: str):
    """Return the storage module matching *backend*.

    Parameters
    ----------
    backend:
        Name of the backend.  Supported values are ``"timescale"`` and
        ``"quest"``.  Any other value defaults to ``quest`` to keep the
        function small and avoid raising additional errors in the workers.
    """

    if backend == "timescale":
        from ..storage import timescale as storage
    else:
        from ..storage import quest as storage
    return storage


async def funding_worker(
    adapter,
    symbol: str,
    *,
    interval: int = 60,
    backend: str = "timescale",
):
    """Periodically fetch funding rates and persist them.

    The worker delegates the retrieval to :func:`data.funding.fetch_funding`
    and stores the normalised result using the selected storage backend.
    """

    from ..data import funding as data_funding

    storage = _get_storage(backend)
    engine = storage.get_engine()

    while True:
        try:
            info = await data_funding.fetch_funding(adapter, symbol)
            storage.insert_funding(
                engine,
                ts=info["ts"],
                exchange=info.get("exchange", getattr(adapter, "name", "unknown")),
                symbol=symbol,
                rate=info.get("rate", 0.0),
                interval_sec=info.get("interval_sec", 0),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - logging only
            log.warning("funding_worker error: %s", exc)
        await asyncio.sleep(interval)


async def open_interest_worker(
    adapter,
    symbol: str,
    *,
    interval: int = 60,
    backend: str = "timescale",
):
    """Periodically fetch open interest and persist it."""

    from ..data import open_interest as data_oi

    storage = _get_storage(backend)
    engine = storage.get_engine()

    while True:
        try:
            info = await data_oi.fetch_oi(adapter, symbol)
            storage.insert_open_interest(
                engine,
                ts=info["ts"],
                exchange=info.get("exchange", getattr(adapter, "name", "unknown")),
                symbol=symbol,
                oi=info.get("oi", 0.0),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - logging only
            log.warning("open_interest_worker error: %s", exc)
        await asyncio.sleep(interval)

