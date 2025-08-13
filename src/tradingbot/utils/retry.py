# src/tradingbot/utils/retry.py
from __future__ import annotations
import asyncio
import random
import logging

log = logging.getLogger(__name__)

def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return any(k in name or k in msg for k in [
        "ratelimit", "network", "timeout", "ddos", "temporarily", "temporarily_unavailable"
    ])

async def with_retry(coro_fn, *args, retries: int = 5, base_delay: float = 0.5, max_delay: float = 4.0, **kwargs):
    """
    Ejecuta una función (sync mediante asyncio.to_thread o ya-async) con backoff.
    Usa: await with_retry(adapter_fn, arg1, arg2, retries=..., base_delay=...)
    """
    attempt = 0
    while True:
        try:
            res = await coro_fn(*args, **kwargs)
            return res
        except Exception as e:
            attempt += 1
            if attempt > retries or not _is_retryable(e):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) * (0.85 + random.random() * 0.3)
            log.warning("retry %d/%d after error: %s (sleep %.2fs)", attempt, retries, e, delay)
            await asyncio.sleep(delay)
