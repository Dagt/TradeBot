from __future__ import annotations

from collections import deque
import bisect
import math
from typing import Deque, Dict, Tuple


class RollingQuantile:
    """Incremental rolling quantile calculator.

    Maintains a fixed-length window of the most recent values. New samples are
    inserted in sorted order using :mod:`bisect` so updates run in ``O(log n)``.
    Quantile queries are then ``O(1)``.  This replaces repeated
    ``pandas.Series.rolling(...).quantile`` calls which require ``O(n)`` work
    per update.
    """

    def __init__(self, window: int, q: float, min_periods: int | None = None) -> None:
        self.window = int(window)
        self.q = float(q)
        self.min_periods = int(min_periods) if min_periods is not None else self.window
        self._values: Deque[float] = deque(maxlen=self.window)
        self._sorted: list[float] = []

    def update(self, value: float) -> float:
        """Insert ``value`` and return the current quantile."""

        if len(self._values) == self.window:
            old = self._values.popleft()
            idx = bisect.bisect_left(self._sorted, old)
            if idx < len(self._sorted):
                self._sorted.pop(idx)
        self._values.append(value)
        bisect.insort(self._sorted, value)

        if len(self._values) < self.min_periods:
            return math.nan

        k = (len(self._sorted) - 1) * self.q
        idx = int(math.floor(k))
        frac = k - idx
        if idx + 1 < len(self._sorted):
            return self._sorted[idx] * (1 - frac) + self._sorted[idx + 1] * frac
        return self._sorted[idx]


class RollingQuantileCache:
    """Cache of :class:`RollingQuantile` objects keyed by ``(symbol, name)``."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], RollingQuantile] = {}

    def get(
        self,
        symbol: str,
        name: str,
        *,
        window: int,
        q: float,
        min_periods: int | None = None,
    ) -> RollingQuantile:
        key = (symbol, name)
        rq = self._cache.get(key)
        if rq is None or rq.window != window or rq.q != q or rq.min_periods != (
            min_periods if min_periods is not None else window
        ):
            rq = RollingQuantile(window, q, min_periods)
            self._cache[key] = rq
        return rq
