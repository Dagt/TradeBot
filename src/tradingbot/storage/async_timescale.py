"""Asynchronous client for TimescaleDB with queued batch writes."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Iterable, Optional

from prometheus_client import Histogram
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from ..config import settings


# Latency of batch writes grouped by table name
BATCH_LATENCY = Histogram(
    "async_storage_batch_latency_seconds",
    "Latency of batch writes to the database",
    ["table"],
)


class AsyncTimescaleClient:
    """Asynchronous SQLAlchemy client with an internal batching queue."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        batch_size: int = 100,
        queue_maxsize: int = 1000,
        flush_interval: float = 1.0,
        insert_sql: Optional[dict[str, str]] = None,
    ) -> None:
        """Create a new client.

        Parameters
        ----------
        dsn:
            Database connection string.  If ``None`` it is built from
            :mod:`tradingbot.config.settings`.
        batch_size:
            Maximum number of rows written per batch.
        queue_maxsize:
            Maximum size of the internal queue before backpressure blocks
            producers.  ``asyncio.Queue`` handles the waiting automatically.
        flush_interval:
            Seconds between automatic flushes of partial batches.
        insert_sql:
            Mapping of table name to ``INSERT`` statement used by
            :meth:`add`.  Additional tables can be registered later via
            :meth:`register_table`.
        """

        if dsn is None:
            dsn = (
                f"postgresql+asyncpg://{settings.pg_user}:{settings.pg_password}"
                f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
            )
        self._dsn = dsn
        self._engine: AsyncEngine | None = None

        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._worker_task: asyncio.Task | None = None
        self._closed = False

        self._insert_sql: dict[str, str] = insert_sql.copy() if insert_sql else {}

    # ------------------------------------------------------------------
    # Connection management
    async def connect(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(self._dsn, pool_pre_ping=True)
        return self._engine

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------
    # Public API
    def register_table(self, table: str, sql: str) -> None:
        """Register ``INSERT`` SQL for *table* used by :meth:`add`."""

        self._insert_sql[table] = sql

    async def add(self, table: str, row: dict[str, Any]) -> None:
        """Queue *row* for asynchronous persistence.

        ``row`` is buffered until ``batch_size`` is reached or ``flush_interval``
        elapses.  Backpressure is handled by ``asyncio.Queue`` which blocks when
        the queue exceeds ``queue_maxsize``.
        """

        if table not in self._insert_sql:
            raise KeyError(f"Unknown table: {table}")
        await self._queue.put((table, row))
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def flush(self) -> None:
        """Flush all pending rows to the database."""

        await self._queue.join()

    async def stop(self) -> None:
        """Flush pending rows and stop the background worker."""

        self._closed = True
        await self.flush()
        if self._worker_task is not None:
            await self._worker_task
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    async def _worker(self) -> None:
        """Background task consuming the queue and writing batches."""

        try:
            while not (self._closed and self._queue.empty()):
                batch: list[tuple[str, dict[str, Any]]] = []
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=self.flush_interval
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass

                while len(batch) < self.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                if not batch:
                    continue

                grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for table, row in batch:
                    grouped[table].append(row)

                for table, rows in grouped.items():
                    sql = self._insert_sql[table]
                    start = time.perf_counter()
                    await self.executemany(sql, rows)
                    BATCH_LATENCY.labels(table).observe(time.perf_counter() - start)

                for _ in batch:
                    self._queue.task_done()
        finally:  # pragma: no cover - worker cleanup
            pass

    # Low level helpers -------------------------------------------------
    async def executemany(self, sql: str, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        engine = await self.connect()
        async with engine.begin() as conn:
            await conn.execute(text(sql), rows)

    async def fetch(self, sql: str) -> list[dict[str, Any]]:
        engine = await self.connect()
        async with engine.connect() as conn:
            res = await conn.execute(text(sql))
            return [dict(r) for r in res.mappings().all()]
