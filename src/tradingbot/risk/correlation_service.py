from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple, Iterable
import math

import pandas as pd


class CorrelationService:
    """Maintain rolling log returns and compute symbol correlations.

    Prices are fed via :meth:`update_price`, which stores log returns
    for each symbol.  :meth:`get_correlations` returns a mapping of
    ``{(sym_a, sym_b): corr}`` for all symbol pairs using data within the
    configured rolling window.
    """

    def __init__(self, window: timedelta | None = None) -> None:
        self.window = window or timedelta(hours=1)
        self._last_price: Dict[str, float] = {}
        self._returns = pd.DataFrame()

    def update_price(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record new ``price`` for ``symbol`` and store its log return."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        # normalise timestamp to second resolution for cross-symbol alignment
        ts = timestamp.replace(microsecond=0)
        prev = self._last_price.get(symbol)
        self._last_price[symbol] = price
        if prev is None or prev <= 0 or price <= 0:
            return
        ret = math.log(price / prev)
        # append return
        self._returns.loc[ts, symbol] = ret
        # trim window
        cutoff = ts - self.window
        if len(self._returns.index):
            self._returns = self._returns.loc[self._returns.index >= cutoff]

    def get_correlations(self) -> Dict[Tuple[str, str], float]:
        """Compute pairwise correlations of returns within the window."""
        if self._returns.empty:
            return {}
        now = datetime.now(timezone.utc)
        recent = self._returns.loc[self._returns.index >= (now - self.window)]
        if recent.shape[0] < 2:
            return {}
        corr = recent.corr()
        result: Dict[Tuple[str, str], float] = {}
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                val = corr.at[a, b]
                if not pd.isna(val):
                    result[(a, b)] = float(val)
        return result

    def purge(self, symbols_active: Iterable[str]) -> None:
        """Remove tracking for symbols not present in ``symbols_active``."""

        active = set(symbols_active)
        for sym in list(self._last_price.keys()):
            if sym not in active:
                self._last_price.pop(sym, None)
        if not self._returns.empty:
            cols = [c for c in self._returns.columns if c in active]
            self._returns = self._returns[cols]
