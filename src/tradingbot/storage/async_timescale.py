"""Asynchronous client for TimescaleDB."""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy import text

from ..config import settings


class AsyncTimescaleClient:
    """Minimal async wrapper around SQLAlchemy for TimescaleDB."""

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            dsn = (
                f"postgresql+asyncpg://{settings.pg_user}:{settings.pg_password}"
                f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"
            )
        self._dsn = dsn
        self._engine: AsyncEngine | None = None

    async def connect(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(self._dsn, pool_pre_ping=True)
        return self._engine

    async def close(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None

    async def executemany(self, sql: str, rows: Iterable[dict[str, Any]]) -> None:
        """Execute *sql* for multiple *rows* in a single transaction."""
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
