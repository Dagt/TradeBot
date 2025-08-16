"""Asynchronous client for QuestDB using PostgreSQL wire protocol."""

from __future__ import annotations

from ..config import settings
from .async_timescale import AsyncTimescaleClient


class AsyncQuestDBClient(AsyncTimescaleClient):
    """QuestDB shares the PostgreSQL protocol so we reuse the Timescale client."""

    def __init__(self, dsn: str | None = None) -> None:
        if dsn is None:
            dsn = (
                f"postgresql+asyncpg://{settings.questdb_user}:{settings.questdb_password}"  # noqa: E501
                f"@{settings.questdb_host}:{settings.questdb_port}/{settings.questdb_db}"  # noqa: E501
            )
        super().__init__(dsn)
