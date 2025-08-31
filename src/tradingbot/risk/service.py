from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

import pandas as pd
import numpy as np

from .exceptions import StopLossExceeded
from .portfolio_guard import PortfolioGuard
from .daily_guard import DailyGuard
from .correlation_service import CorrelationService
from ..core import Account as CoreAccount, RiskManager as CoreRiskManager
from ..storage import timescale
from ..utils.logging import get_logger

log = get_logger(__name__)

@dataclass
class Position:
    qty: float = 0.0
    realized_pnl: float = 0.0

class _RiskManager:
    def __init__(self, risk_pct: float = 0.0, *, allow_short: bool = True, min_order_qty: float = 1e-9):
        self.risk_pct = abs(risk_pct)
        self.allow_short = bool(allow_short)
        self.min_order_qty = float(min_order_qty)
        self.enabled = True
        self.last_kill_reason: str | None = None
        self.pos = Position()
        self.positions_multi: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._entry_price: float | None = None

    def update_position(self, exchange: str, symbol: str, qty: float) -> None:
        book = self.positions_multi.setdefault(exchange, {})
        book[symbol] = float(qty)

    def aggregate_positions(self) -> Dict[str, float]:
        agg: Dict[str, float] = {}
        for book in self.positions_multi.values():
            for sym, qty in book.items():
                agg[sym] = agg.get(sym, 0.0) + float(qty)
        return agg

    def _reset_price_trackers(self) -> None:
        self._entry_price = None

    def add_fill(self, side: str, qty: float, price: float | None = None) -> None:
        signed = float(qty) if side == 'buy' else -float(qty)
        prev = self.pos.qty
        if prev * signed < 0 and price is not None and self._entry_price is not None:
            closed = min(abs(prev), abs(signed))
            pnl = (float(price) - self._entry_price) * closed
            if prev < 0:
                pnl = -pnl
            self.pos.realized_pnl += pnl
        self.pos.qty += signed
        new = self.pos.qty
        if prev == 0 or prev * new <= 0:
            self._reset_price_trackers()
            if price is not None and abs(new) > 0:
                self._entry_price = float(price)
            return
        if prev * signed < 0:
            return
        if price is not None:
            abs_prev = abs(prev)
            abs_new = abs(signed)
            if self._entry_price is None:
                self._entry_price = float(price)
            else:
                self._entry_price = (self._entry_price * abs_prev + float(price) * abs_new) / (abs_prev + abs_new)

    def check_limits(self, price: float) -> bool:
        if not self.enabled:
            return False
        qty = self.pos.qty
        if abs(qty) < 1e-12:
            self._reset_price_trackers()
            return True
        px = float(price)
        if self._entry_price is None:
            self._entry_price = px
            return True
        notional = abs(qty) * self._entry_price
        pnl = (px - self._entry_price) * qty
        if self.risk_pct > 0 and pnl < -notional * self.risk_pct:
            self.last_kill_reason = 'stop_loss'
            raise StopLossExceeded('stop_loss')
        return True

    def update_correlation(self, symbol_pairs: Dict[Tuple[str, str], float], threshold: float) -> List[Tuple[str, str]]:
        return [pair for pair, corr in symbol_pairs.items() if abs(corr) >= threshold]

    def update_covariance(self, cov_matrix: Dict[Tuple[str, str], float], threshold: float) -> List[Tuple[str, str]]:
        exceeded: List[Tuple[str, str]] = []
        vars: Dict[str, float] = {}
        for (a, b), cov in cov_matrix.items():
            if a == b:
                vars[a] = abs(cov)
        for (a, b), cov in cov_matrix.items():
            if a == b:
                continue
            va = vars.get(a)
            vb = vars.get(b)
            if va and vb and va > 0 and vb > 0:
                corr = cov / ((va ** 0.5) * (vb ** 0.5))
                if abs(corr) >= threshold:
                    exceeded.append((a, b))
        return exceeded

    def reset(self) -> None:
        self.enabled = True
        self.last_kill_reason = None
        self.pos.qty = 0.0
        self.positions_multi.clear()
        self._reset_price_trackers()

    def register_order(self, notional: float) -> bool:
        return True

    def complete_order(self) -> None:
        return None



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
        guard: PortfolioGuard,
        daily: DailyGuard | None = None,
        corr_service: CorrelationService | None = None,
        *,
        bus=None,
        engine=None,
        account: CoreAccount | None = None,
        risk_per_trade: float = 0.01,
        atr_mult: float = 2.0,
        risk_pct: float = 0.01,
    ) -> None:
        self.rm = _RiskManager(risk_pct=risk_pct)
        self.guard = guard
        self.daily = daily
        self.corr = corr_service
        self.engine = engine
        self.bus = bus
        self.account = account or CoreAccount(float("inf"))
        self.core = CoreRiskManager(
            self.account,
            risk_per_trade=risk_per_trade,
            atr_mult=atr_mult,
            risk_pct=risk_pct,
        )
        self.trades: Dict[str, dict] = {}

    @property
    def min_order_qty(self) -> float:
        return self.rm.min_order_qty

    @property
    def allow_short(self) -> bool:
        """Whether short positions are permitted."""
        return self.rm.allow_short

    @allow_short.setter
    def allow_short(self, value: bool) -> None:
        self.rm.allow_short = bool(value)

    @property
    def pos(self) -> Position:
        """Current aggregate position."""
        return self.rm.pos

    @property
    def positions_multi(self) -> Dict[str, Dict[str, float]]:
        """Per-venue positions bookkeeping."""
        return self.rm.positions_multi

    def register_order(self, notional: float) -> bool:
        return self.rm.register_order(notional)

    def complete_order(self) -> None:
        self.rm.complete_order()

    def set_position(self, qty: float) -> None:
        self.rm.pos.qty = float(qty)

    def reset(self) -> None:
        """Reset underlying risk manager state."""
        self.rm.reset()


    # ------------------------------------------------------------------
    # Position helpers
    def update_position(
        self, exchange: str, symbol: str, qty: float, *, entry_price: float | None = None
    ) -> None:
        """Synchronise position and track trade state for ``symbol``."""
        self.rm.update_position(exchange, symbol, qty)
        self.guard.set_position(exchange, symbol, qty)
        cur = self.account.positions.get(symbol, 0.0)
        self.account.update_position(symbol, qty - cur, price=entry_price)
        if abs(qty) < self.rm.min_order_qty:
            self.trades.pop(symbol, None)
            return
        side = "buy" if qty > 0 else "sell"
        trade = self.trades.get(symbol, {})
        trade.update({"side": side, "qty": abs(qty)})
        if entry_price is not None:
            trade["entry_price"] = entry_price
            trade.setdefault("stop", entry_price)
        trade.setdefault("stage", 0)
        self.trades[symbol] = trade

    def aggregate_positions(self) -> Dict[str, float]:
        """Return aggregated positions across all venues."""
        return self.rm.aggregate_positions()

    def get_trade(self, symbol: str) -> dict | None:
        """Return tracked trade for ``symbol`` if any."""
        return self.trades.get(symbol)

    # Delegates to core risk manager
    def calc_position_size(self, signal_strength: float, price: float) -> float:
        return self.core.calc_position_size(signal_strength, price)

    def check_global_exposure(self, symbol: str, new_alloc: float) -> bool:
        current = self.account.current_exposure(symbol)[1]
        pending = self.account.pending_exposure(symbol)
        limit = float(self.account.max_symbol_exposure)
        return current + pending + abs(float(new_alloc)) <= limit

    def initial_stop(
        self, entry_price: float, side: str, atr: float | None = None
    ) -> float:
        return self.core.initial_stop(entry_price, side, atr)

    def update_trailing(
        self, trade: dict | object, current_price: float, fees_slip: float = 0.0
    ) -> None:
        self.core.update_trailing(trade, current_price, fees_slip)

    def manage_position(self, trade: dict | object, signal: dict | None = None) -> str:
        return self.core.manage_position(trade, signal)

    # ------------------------------------------------------------------
    # Price tracking and risk checks
    def mark_price(self, symbol: str, price: float) -> None:
        self.guard.mark_price(symbol, price)
        self.account.mark_price(symbol, price)
        if self.corr is not None:
            self.corr.update_price(symbol, price)

    def check_order(
        self,
        symbol: str,
        side: str,
        price: float,
        strength: float = 1.0,
        *,
        corr_threshold: float = 0.0,
        pending_qty: float = 0.0,
    ) -> tuple[bool, str, float]:
        """Check limits and compute sized order before submitting.

        Returns ``(allowed, reason, delta)`` where ``delta`` is the signed size
        based solely on ``signal_strength`` and ``price``.
        """

        # refresh caps based on current account cash
        self.guard.refresh_usd_caps(self.account.cash)

        corr_pairs: Dict[tuple[str, str], float] = {}
        if self.corr is not None:
            corr_pairs = self.corr.get_correlations()

        unsigned = self.calc_position_size(strength, price)
        delta = unsigned if side.lower() == "buy" else -unsigned

        if pending_qty > 0:
            if delta > 0:
                delta = max(delta - pending_qty, 0.0)
            elif delta < 0:
                delta = min(delta + pending_qty, 0.0)

        qty = abs(delta)
        alloc = qty * price
        if not self.check_global_exposure(symbol, alloc):
            return False, "symbol_exposure", 0.0

        if qty < self.rm.min_order_qty:
            return False, "zero_size", 0.0

        try:
            limits_ok = self.rm.check_limits(price)
        except StopLossExceeded:
            self._persist("VIOLATION", symbol, "stop_loss", {})
            cur = (
                self.guard.st.positions.get(symbol, self.rm.pos.qty)
                if self.guard is not None
                else self.rm.pos.qty
            )
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
        self.account.update_open_order(symbol, qty)
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
        self.account.update_open_order(symbol, -abs(qty))
        delta = qty if side == "buy" else -qty
        self.account.update_position(symbol, delta, price=price)
        if venue is not None:
            book = self.guard.st.venue_positions.get(venue, {})
            self.rm.update_position(venue, symbol, book.get(symbol, 0.0))
        cur_qty, _ = self.account.current_exposure(symbol)
        if abs(cur_qty) < self.rm.min_order_qty:
            self.trades.pop(symbol, None)
            return
        trade = self.trades.get(symbol, {})
        trade.update(
            {
                "side": "buy" if cur_qty > 0 else "sell",
                "qty": abs(cur_qty),
                "entry_price": getattr(self.rm, "_entry_price", None),
            }
        )
        trade.setdefault("stop", trade["entry_price"])
        trade.setdefault("stage", 0)
        self.trades[symbol] = trade

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


# ---------------------------------------------------------------------------
# Rehydration helpers


def load_positions(engine, venue: str) -> dict:
    """Load persisted positions for ``venue`` from storage."""
    if engine is None:
        return {}
    try:
        from ..storage.timescale import load_positions as _load  # type: ignore
        return _load(engine, venue)
    except Exception:
        try:
            with engine.begin() as conn:
                rows = conn.execute(
                    text(
                        'SELECT venue, symbol, qty, avg_price, realized_pnl, fees_paid '
                        'FROM "market.positions" WHERE venue = :venue'
                    ),
                    {"venue": venue},
                ).mappings().all()
            out = {}
            for r in rows:
                out[r["symbol"]] = {
                    "qty": float(r["qty"]),
                    "avg_price": float(r["avg_price"]),
                    "realized_pnl": float(r["realized_pnl"]),
                    "fees_paid": float(r["fees_paid"]),
                }
            return out
        except Exception:
            return {}
