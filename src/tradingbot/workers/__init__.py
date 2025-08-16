"""Background workers used by the bot."""

from .ingestion import BatchIngestionWorker

__all__ = ["BatchIngestionWorker"]
