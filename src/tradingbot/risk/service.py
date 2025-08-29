from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Tuple

import pandas as pd
import numpy as np

from .manager import RiskManager
from .exceptions import StopLossExceeded
from .portfolio_guard import PortfolioGuard
from .daily_guard import DailyGuard
from .correlation_service import CorrelationService
from ..storage import timescale
from ..utils.logging import get_logger

log = get_logger(__name__)


class RiskService:
    """Facade combining risk components.

    It keeps ``RiskManager``, ``PortfolioGuard`` and ``DailyGuard`` wired
    together and exposes a small public API used by the runners.
    ``update_position`` keeps the multi-venue book in sync while
    ``aggregate_positions`` returns the consolidated exposures.
    Risk events are optionally persisted to TimescaleDB.
    """

    def __init__(
        self,
        manager: RiskManager,
        guard: PortfolioGuard,
        daily: DailyGuard | None = None,
        corr_service: CorrelationService | None = None,
        *,
        engine=None,
    ) -> None:
        self.rm = manager
        self.guard = guard
        self.daily = daily
        self.corr = corr_service
        self.engine = engine

    # ------------------------------------------------------------------
    # Position helpers
    def update_position(self, exchange: str, symbol: str, qty: float) -> None:
        """Synchronise position for ``exchange``/``symbol`` across components."""
        self.rm.update_position(exchange, symbol, qty)
        self.guard.set_position(exchange, symbol, qty)

    def aggregate_positions(self) -> Dict[str, float]:
        """Return aggregated positions across all venues."""
        return self.rm.aggregate_positions()

    # ------------------------------------------------------------------
    # Price tracking and risk checks
    def mark_price(self, symbol: str, price: float) -> None:
        self.guard.mark_price(symbol, price)
        if self.corr is not None:
            self.corr.update_price(symbol, price)

    def check_order(
        self,
        symbol: str,
        side: str,
        equity: float,
        price: float,
        strength: float = 1.0,
        *,
        symbol_vol: float | None = None,
        corr_threshold: float = 0.0,
    ) -> tuple[bool, str, float]:
        """Check limits and compute sized order before submitting.

        Returns ``(allowed, reason, delta)`` where ``delta`` is the signed size
        proposed after volatility and correlation adjustments.
        """

        self.guard.refresh_usd_caps(equity)

        if symbol_vol is None or symbol_vol <= 0:
            symbol_vol = self.guard.volatility(symbol)

        correlations: Dict[Tuple[str, str], float] = {}
        if self.corr is not None:
            correlations = self.corr.get_correlations()

        delta = self.rm.size(
            side,
            strength=strength,
            equity=equity,
            price=price,
            symbol=symbol,
            symbol_vol=symbol_vol or 0.0,
            correlations=correlations,
            threshold=corr_threshold,
        )
        qty = abs(delta)

        if qty < self.rm.min_order_qty:
            return False, "zero_size", 0.0

        try:
            limits_ok = self.rm.check_limits(price)
        except StopLossExceeded:
            self._persist("VIOLATION", symbol, "stop_loss", {})
            cur = self.guard.st.positions.get(symbol, 0.0)
            if abs(cur) > 0:
                return True, "stop_loss", -cur
            return False, "stop_loss", 0.0

        if not limits_ok:
            reason = self.rm.last_kill_reason or "kill_switch"
            self._persist("VIOLATION", symbol, reason, {})
            return False, reason, 0.0

        action, reason, metrics = self.guard.soft_cap_decision(symbol, side, qty, price)
        if action == "block":
            self._persist("VIOLATION", symbol, reason, metrics)
            return False, reason, delta
        if action == "soft_allow":
            self._persist("INFO", symbol, reason, metrics)
        return True, reason, delta

    # ------------------------------------------------------------------
    def update_correlation(
        self, corr_df: pd.DataFrame | np.ndarray, threshold: float
    ) -> list[tuple[str, str]]:
        """Apply correlation limits from a matrix of correlations.

        ``corr_df`` may be a :class:`pandas.DataFrame` or a 2D array-like
        object.  Only pairs whose absolute correlation exceeds ``threshold``
        are forwarded to :class:`RiskManager`.
        """

        if isinstance(corr_df, np.ndarray):
            corr_df = pd.DataFrame(corr_df)

        if not isinstance(corr_df, pd.DataFrame):
            return []

        syms = list(corr_df.columns)
        pairs: Dict[tuple[str, str], float] = {}
        for idx, a in enumerate(syms):
            for b in syms[idx + 1 :]:
                val = corr_df.at[a, b]
                if pd.isna(val):
                    continue
                if abs(float(val)) >= threshold:
                    pairs[(a, b)] = float(val)

        if not pairs:
            return []
        return self.rm.update_correlation(pairs, threshold)

    def update_covariance(
        self, cov_df: pd.DataFrame | np.ndarray, threshold: float
    ) -> list[tuple[str, str]]:
        """Apply covariance-based correlation limits.

        ``cov_df`` may be a covariance matrix provided as DataFrame or array.
        Off-diagonal pairs are kept only if their implied correlation exceeds
        ``threshold``.  Diagonal elements are always forwarded so that
        :class:`RiskManager` can compute the correlations.
        """

        if isinstance(cov_df, np.ndarray):
            cov_df = pd.DataFrame(cov_df)

        if not isinstance(cov_df, pd.DataFrame):
            return []

        syms = list(cov_df.columns)
        cov_matrix: Dict[tuple[str, str], float] = {}
        vars: Dict[str, float] = {}

        for s in syms:
            val = cov_df.at[s, s]
            if not pd.isna(val):
                vars[s] = float(abs(val))
                cov_matrix[(s, s)] = float(val)

        for i, a in enumerate(syms):
            for b in syms[i + 1 :]:
                val = cov_df.at[a, b]
                if pd.isna(val):
                    continue
                va = vars.get(a)
                vb = vars.get(b)
                if va and vb and va > 0 and vb > 0:
                    corr = float(val) / ((va ** 0.5) * (vb ** 0.5))
                    if abs(corr) >= threshold:
                        cov_matrix[(a, b)] = float(val)

        if not cov_matrix:
            return []
        return self.rm.update_covariance(cov_matrix, threshold)

    # ------------------------------------------------------------------
    # Fill / PnL updates
    def on_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        venue: str | None = None,
    ) -> None:
        """Update internal position books after a fill."""
        self.rm.add_fill(side, qty, price=price)
        self.guard.update_position_on_order(symbol, side, qty, venue=venue)
        if venue is not None:
            book = self.guard.st.venue_positions.get(venue, {})
            self.rm.update_position(venue, symbol, book.get(symbol, 0.0))

    def daily_mark(self, broker, symbol: str, price: float, delta_rpnl: float) -> tuple[bool, str]:
        """Update daily guard with latest mark and realized PnL delta."""
        if self.daily is None:
            return False, ""
        now = datetime.now(timezone.utc)
        self.daily.on_mark(now, equity_now=broker.equity(mark_prices={symbol: price}))
        if abs(delta_rpnl) > 0:
            self.daily.on_realized_delta(delta_rpnl)
        halted, reason = self.daily.check_halt(broker)
        if halted:
            self._persist(f"HALT_{reason}", symbol, f"HALT: {reason}", {})
        return halted, reason

    # ------------------------------------------------------------------
    def _persist(self, kind: str, symbol: str, message: str, details: dict) -> None:
        if self.engine is None:
            return
        try:
            timescale.insert_risk_event(
                self.engine,
                venue=self.guard.cfg.venue,
                symbol=symbol,
                kind=kind,
                message=message,
                details=details,
            )
        except Exception:  # pragma: no cover - persistence shouldn't break tests
            log.exception("Persist failure: risk_event %s", kind)
