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
