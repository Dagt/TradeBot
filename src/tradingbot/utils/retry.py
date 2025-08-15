# src/tradingbot/utils/retry.py
from __future__ import annotations

import asyncio
import inspect
import logging
import random

log = logging.getLogger(__name__)

def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    msg = str(exc).lower()
    return any(k in name or k in msg for k in [
        "ratelimit", "network", "timeout", "ddos", "temporarily", "temporarily_unavailable"
    ])

async def _run_fn(fn, *args, **kwargs):
    """Run ``fn`` regardless of it being sync or async.

    Synchronous callables are executed in a thread via ``asyncio.to_thread`` to
    avoid blocking the event loop.  Async callables are awaited directly.  If a
    synchronous callable returns an awaitable (e.g. a partial of an async
    function), the awaitable is awaited before returning the final result.
    """

    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)

    res = await asyncio.to_thread(fn, *args, **kwargs)
    if inspect.isawaitable(res):
        return await res
    return res


async def with_retry(fn, *args, retries: int = 5, base_delay: float = 0.5,
                     max_delay: float = 4.0, **kwargs):
    """Execute ``fn`` with exponential backoff retries.

    ``fn`` may be either a synchronous or asynchronous callable.  For
    synchronous callables the execution is dispatched to ``asyncio.to_thread``.
    """
    attempt = 0
    while True:
        try:
            return await _run_fn(fn, *args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > retries or not _is_retryable(e):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= 0.85 + random.random() * 0.3
            log.warning(
                "retry %d/%d after error: %s (sleep %.2fs)",
                attempt,
                retries,
                e,
                delay,
            )
            await asyncio.sleep(delay)
