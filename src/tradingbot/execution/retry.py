# src/tradingbot/execution/retry.py
from __future__ import annotations
import asyncio, random, time, logging
from typing import Callable, Awaitable, Any, Optional

log = logging.getLogger(__name__)

class AmbiguousOrderError(Exception):
    """La orden pudo haber sido aceptada/ejecutada, pero el cliente recibió error/timeout."""
    def __init__(self, message: str, candidate_client_order_id: Optional[str] = None):
        super().__init__(message)
        self.client_order_id = candidate_client_order_id

def _sleep_backoff(attempt: int, base: float = 0.25, cap: float = 3.0):
    # backoff exponencial con jitter
    delay = min(cap, base * (2 ** attempt))
    delay = delay * (0.6 + 0.8 * random.random())
    return delay

async def with_retries(async_fn: Callable[[], Awaitable[Any]], *,
                       max_attempts: int = 5,
                       on_retry: Optional[Callable[[int, BaseException], None]] = None):
    """
    Ejecuta async_fn con reintentos y backoff. Si lanza AmbiguousOrderError,
    deja propagar para que el caller haga reconciliación.
    """
    attempt = 0
    while True:
        try:
            return await async_fn()
        except AmbiguousOrderError:
            # Se maneja afuera para reconciliar
            raise
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise
            if on_retry:
                on_retry(attempt, e)
            await asyncio.sleep(_sleep_backoff(attempt))
