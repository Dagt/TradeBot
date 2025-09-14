from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from threading import Lock
import time

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
        profit_lock_usd: float = 1.0,
        market_type: str | None = None,
    ) -> None:
        self.guard = guard
        self.daily = daily
        self.corr = corr_service
        self.engine = engine
        self.bus = bus
        mt = market_type or getattr(account, "market_type", None) or "futures"
        self.account = account or CoreAccount(float("inf"), market_type=mt)
        if account is not None:
            account.market_type = mt
        self.rm = CoreRiskManager(
            self.account,
            risk_per_trade=risk_per_trade,
            atr_mult=atr_mult,
            risk_pct=risk_pct,
            profit_lock_usd=profit_lock_usd,
        )
        self.trades: Dict[str, dict] = {}

        # Internal risk tracking previously handled by _RiskManager
        self.risk_pct = abs(risk_pct)
        self.market_type = mt
        self._allow_short = mt != "spot"
        self._min_order_qty = 1e-9
        self._min_notional = 0.0
        self.enabled = True
        self.last_kill_reason: str | None = None
        self._pos = Position()
        self.positions_multi: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._entry_price: float | None = None
        self._lock = Lock()

    @property
    def min_order_qty(self) -> float:
        return self._min_order_qty

    @min_order_qty.setter
    def min_order_qty(self, value: float) -> None:
        self._min_order_qty = float(value)

    @property
    def min_notional(self) -> float:
        return self._min_notional

    @min_notional.setter
    def min_notional(self, value: float) -> None:
        self._min_notional = float(value)

    @property
    def allow_short(self) -> bool:
        """Whether short positions are permitted."""
        return self._allow_short

    @allow_short.setter
    def allow_short(self, value: bool) -> None:
        self._allow_short = bool(value)

    @property
    def pos(self) -> Position:
        """Current aggregate position."""
        return self._pos

    def register_order(self, symbol: str, notional: float) -> bool:
        """Check projected exposure for ``symbol`` before queuing an order.

        The method considers any outstanding open orders via
        :meth:`Account.pending_exposure` and compares the projected
        exposure against the hard caps enforced by :class:`PortfolioGuard`
        and the state of :class:`DailyGuard`.
        """

        if not self.enabled:
            return False

        pending_existing = self.account.pending_exposure(symbol)
        pending_other = sum(
            self.account.pending_exposure(sym)
            for sym in self.account.open_orders
            if sym != symbol
        )
        proj_sym_exp = (
            self.guard.exposure_symbol(symbol)
            + pending_existing
            + abs(float(notional))
        )
        proj_total_exp = (
            self.guard.exposure_total()
            + pending_other
            + pending_existing
            + abs(float(notional))
        )

        per_cap = getattr(self.guard.st, "per_symbol_cap_usdt", None)
        if per_cap is not None and proj_sym_exp > per_cap:
            reason = (
                f"per_symbol_cap_usdt excedido ({proj_sym_exp:.2f} > {per_cap:.2f})"
            )
            self.last_kill_reason = reason
            self._persist(
                "VIOLATION", symbol, reason, {"sym_exp": proj_sym_exp, "total": proj_total_exp}
            )
            return False

        tot_cap = getattr(self.guard.st, "total_cap_usdt", None)
        if tot_cap is not None and proj_total_exp > tot_cap:
            reason = f"total_cap_usdt excedido ({proj_total_exp:.2f} > {tot_cap:.2f})"
            self.last_kill_reason = reason
            self._persist(
                "VIOLATION", symbol, reason, {"sym_exp": proj_sym_exp, "total": proj_total_exp}
            )
            return False

        if self.daily is not None and getattr(self.daily, "halted", False):
            reason = "daily_halt"
            self.last_kill_reason = reason
            self._persist("VIOLATION", symbol, reason, {})
            return False

        return True

    def complete_order(self) -> None:
        return None

    def set_position(self, qty: float) -> None:
        self._pos.qty = float(qty)

    def reset(self) -> None:
        """Reset internal risk state."""
        with self._lock:
            self.enabled = True
            self.last_kill_reason = None
            self._pos.qty = 0.0
            self.positions_multi.clear()
            self._reset_price_trackers()

    # ------------------------------------------------------------------
    # Internal helpers replacing former _RiskManager methods
    def _reset_price_trackers(self) -> None:
        self._entry_price = None

    def add_fill(self, side: str, qty: float, price: float | None = None) -> None:
        signed = float(qty) if side == "buy" else -float(qty)
        prev = self._pos.qty
        if prev * signed < 0 and price is not None and self._entry_price is not None:
            closed = min(abs(prev), abs(signed))
            pnl = (float(price) - self._entry_price) * closed
            if prev < 0:
                pnl = -pnl
            self._pos.realized_pnl += pnl
        self._pos.qty += signed
        new = self._pos.qty
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
                self._entry_price = (
                    self._entry_price * abs_prev + float(price) * abs_new
                ) / (abs_prev + abs_new)

    def check_limits(self, price: float) -> bool:
        if not self.enabled:
            return False
        qty = self._pos.qty
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
            self.last_kill_reason = "stop_loss"
            raise StopLossExceeded("stop_loss")
        return True


    # ------------------------------------------------------------------
    # Position helpers
    def update_position(
        self,
        exchange: str,
        symbol: str,
        qty: float,
        *,
        entry_price: float | None = None,
        atr: float | None = None,
    ) -> None:
        """Synchronise position and track trade state for ``symbol``."""
        with self._lock:
            book = self.positions_multi.setdefault(exchange, {})
            book[symbol] = float(qty)
            self.guard.set_position(exchange, symbol, qty)
            cur = self.account.positions.get(symbol, 0.0)
            self.account.update_position(symbol, qty - cur, price=entry_price)
            if abs(qty) < self._min_order_qty:
                self.trades.pop(symbol, None)
                return
            side = "buy" if qty > 0 else "sell"
            is_new_trade = symbol not in self.trades
            trade = self.trades.get(symbol, {})
            trade.update({"side": side, "qty": abs(qty)})
            if atr is not None:
                trade["atr"] = float(atr)
            if entry_price is not None:
                trade["entry_price"] = entry_price
                trade.setdefault("stop", self.initial_stop(entry_price, side, atr))
            if is_new_trade:
                trade["stage"] = 0
            self.trades[symbol] = trade

    def aggregate_positions(self) -> Dict[str, float]:
        """Return aggregated positions across all venues."""
        agg: Dict[str, float] = {}
        for book in self.positions_multi.values():
            for sym, qty in book.items():
                agg[sym] = agg.get(sym, 0.0) + float(qty)
        return agg

    def purge(self, symbols_active: Iterable[str]) -> None:
        """Remove bookkeeping entries for inactive symbols.

        Parameters
        ----------
        symbols_active:
            Iterable of symbols that should be retained. Any symbol not present
            here will be removed from internal dictionaries to keep the risk
            state tidy.
        """

        active = set(symbols_active)

        with self._lock:
            # trades
            for sym in list(self.trades.keys()):
                if sym not in active:
                    self.trades.pop(sym, None)

            # multi-venue positions
            for venue, book in list(self.positions_multi.items()):
                for sym in list(book.keys()):
                    if sym not in active:
                        book.pop(sym, None)
                if not book:
                    self.positions_multi.pop(venue, None)

            # account level tracking
            for mapping in (
                self.account.positions,
                self.account.prices,
                self.account.open_orders,
            ):
                for sym in list(mapping.keys()):
                    if sym not in active:
                        mapping.pop(sym, None)

        # portfolio guard state
        st = getattr(self.guard, "st", None)
        if st is not None:
            for attr in ("positions", "prices", "returns", "sym_soft_started"):
                data = getattr(st, attr, None)
                if isinstance(data, dict):
                    for sym in list(data.keys()):
                        if sym not in active:
                            data.pop(sym, None)
            # per-venue positions
            vp = getattr(st, "venue_positions", None)
            if isinstance(vp, dict):
                for venue, book in list(vp.items()):
                    for sym in list(book.keys()):
                        if sym not in active:
                            book.pop(sym, None)
                    if not book:
                        vp.pop(venue, None)

    def get_trade(self, symbol: str) -> dict | None:
        """Return tracked trade for ``symbol`` if any."""
        return self.trades.get(symbol)

    # Delegates to core risk manager
    def calc_position_size(
        self,
        signal_strength: float,
        price: float,
        *,
        volatility: float | None = None,
        target_volatility: float | None = None,
        clamp: bool = True,
    ) -> float:
        return self.rm.calc_position_size(
            signal_strength,
            price,
            volatility=volatility,
            target_volatility=target_volatility,
            clamp=clamp,
        )

    def check_global_exposure(self, symbol: str, new_alloc: float) -> bool:
        return self.rm.check_global_exposure(symbol, new_alloc)

    def initial_stop(
        self,
        entry_price: float,
        side: str,
        atr: float | None = None,
        *,
        atr_mult: float | None = None,
    ) -> float:
        return self.rm.initial_stop(entry_price, side, atr, atr_mult=atr_mult)

    def update_trailing(
        self,
        trade: dict | object,
        current_price: float,
        fees_slip: float = 0.0,
        profit_lock_usd: float | None = None,
    ) -> None:
        self.rm.update_trailing(
            trade,
            current_price,
            fees_slip,
            profit_lock_usd=profit_lock_usd,
        )

    def manage_position(self, trade: dict | object, signal: dict | None = None) -> str:
        price = None
        if isinstance(trade, dict):
            price = trade.get("current_price")
            trail_done = trade.get("_trail_done")
            trade["bars_held"] = int(trade.get("bars_held", 0)) + 1
            max_hold = trade.get("max_hold")
            if (
                max_hold is not None
                and int(trade["bars_held"]) >= int(max_hold)
                and price is not None
            ):
                entry = trade.get("entry_price")
                atr = trade.get("atr")
                side = trade.get("side")
                if entry is not None and atr is not None and side is not None:
                    move = (
                        float(price) - float(entry)
                        if side.lower() == "buy"
                        else float(entry) - float(price)
                    )
                    if move < 0.5 * float(atr):
                        return "close"
        else:
            price = getattr(trade, "current_price", None)
            trail_done = getattr(trade, "_trail_done", False)
            bars = int(getattr(trade, "bars_held", 0)) + 1
            setattr(trade, "bars_held", bars)
            max_hold = getattr(trade, "max_hold", None)
            if max_hold is not None and bars >= int(max_hold) and price is not None:
                entry = getattr(trade, "entry_price", None)
                atr = getattr(trade, "atr", None)
                side = getattr(trade, "side", None)
                if entry is not None and atr is not None and side is not None:
                    move = (
                        float(price) - float(entry)
                        if str(side).lower() == "buy"
                        else float(entry) - float(price)
                    )
                    if move < 0.5 * float(atr):
                        return "close"
        if price is not None and not trail_done:
            self.update_trailing(trade, float(price))
            if isinstance(trade, dict):
                trade["_trail_done"] = True
            else:
                setattr(trade, "_trail_done", True)
        return self.rm.manage_position(trade, signal)

    # ------------------------------------------------------------------
    # Price tracking and risk checks
    def mark_price(self, symbol: str, price: float) -> None:
        with self._lock:
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
        volatility: float | None = None,
        target_volatility: float | None = None,
    ) -> tuple[bool, str, float]:
        """Check limits and compute sized order before submitting.

        Parameters
        ----------
        pending_qty:
            Quantity already reserved by open orders for ``symbol``.  This
            amount is subtracted from the computed ``delta`` so that subsequent
            orders account for outstanding amounts.
        volatility:
            Current volatility measure (e.g. ATR) used to scale risk.
        target_volatility:
            Desired volatility level.  Position size is scaled by
            ``target_volatility / volatility`` when both values are supplied.

        Returns
        -------
        tuple[bool, str, float]
            ``(allowed, reason, delta)`` where ``delta`` is the signed size
            based solely on ``signal_strength`` and ``price``.
        """

        # refresh caps based on current account cash
        self.guard.refresh_usd_caps(self.account.cash)

        corr_pairs: Dict[tuple[str, str], float] = {}
        if self.corr is not None:
            corr_pairs = self.corr.get_correlations()

        unsigned = self.calc_position_size(
            strength,
            price,
            volatility=volatility,
            target_volatility=target_volatility,
            clamp=False,
        )
        delta = unsigned if side.lower() == "buy" else -unsigned

        if pending_qty > 0:
            if delta > 0:
                delta = max(delta - pending_qty, 0.0)
            elif delta < 0:
                delta = min(delta + pending_qty, 0.0)

        if delta < 0 and not self.allow_short:
            cur_qty, _ = self.account.current_exposure(symbol)
            if cur_qty <= 0:
                log.warning("short_not_allowed: %s cur_qty=%.8f", symbol, cur_qty)
                return False, "short_not_allowed", 0.0
            sellable = min(abs(delta), cur_qty)
            delta = -sellable

        qty = abs(delta)
        alloc = qty * price
        if not self.check_global_exposure(symbol, alloc):
            return False, "symbol_exposure", 0.0

        if qty < self._min_order_qty:
            return False, "below_min_qty", 0.0

        try:
            limits_ok = self.check_limits(price)
        except StopLossExceeded:
            self._persist("VIOLATION", symbol, "stop_loss", {})
            cur = (
                self.guard.st.positions.get(symbol, self._pos.qty)
                if self.guard is not None
                else self._pos.qty
            )
            if abs(cur) > 0:
                return True, "stop_loss", -cur
            return False, "stop_loss", 0.0

        if not limits_ok:
            reason = self.last_kill_reason or "kill_switch"
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
        self,
        corr_df: pd.DataFrame | np.ndarray | Dict[tuple[str, str], float],
        threshold: float,
    ) -> list[tuple[str, str]]:
        """Return symbol pairs whose correlation exceeds ``threshold``."""

        if isinstance(corr_df, dict):
            return [pair for pair, corr in corr_df.items() if abs(float(corr)) >= threshold]

        if isinstance(corr_df, np.ndarray):
            corr_df = pd.DataFrame(corr_df)

        if not isinstance(corr_df, pd.DataFrame):
            return []

        syms = list(corr_df.columns)
        exceeded: list[tuple[str, str]] = []
        for idx, a in enumerate(syms):
            for b in syms[idx + 1 :]:
                val = corr_df.at[a, b]
                if pd.isna(val):
                    continue
                if abs(float(val)) >= threshold:
                    exceeded.append((a, b))
        return exceeded

    def update_covariance(
        self,
        cov_df: pd.DataFrame | np.ndarray | Dict[tuple[str, str], float],
        threshold: float,
    ) -> list[tuple[str, str]]:
        """Return pairs whose implied correlation from covariance exceeds ``threshold``."""

        if isinstance(cov_df, dict):
            cov_matrix = cov_df
        else:
            if isinstance(cov_df, np.ndarray):
                cov_df = pd.DataFrame(cov_df)
            if not isinstance(cov_df, pd.DataFrame):
                return []
            syms = list(cov_df.columns)
            cov_matrix = {}
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

        exceeded: list[tuple[str, str]] = []
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

    def covariance_matrix(self, returns: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
        """Compute covariance matrix from return series."""
        if not returns:
            return {}
        syms = list(returns.keys())
        arr = np.array([returns[s] for s in syms])
        cov = np.cov(arr, ddof=1)
        out: Dict[Tuple[str, str], float] = {}
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                out[(a, b)] = float(cov[i, j])
        return out

    def adjust_size_for_correlation(
        self,
        symbol: str,
        size: float,
        correlations: Dict[Tuple[str, str], float],
        threshold: float,
    ) -> float:
        """Reduce ``size`` when ``symbol`` is highly correlated with others."""
        for (a, b), corr in correlations.items():
            if symbol in (a, b) and abs(float(corr)) >= threshold:
                return size / 2.0
        return size

    def check_portfolio_risk(
        self,
        allocations: Dict[str, float],
        cov_matrix: Dict[Tuple[str, str], float],
        limit: float,
    ) -> bool:
        """Return ``True`` if portfolio variance does not exceed ``limit``."""
        var = 0.0
        syms = list(allocations.keys())
        for i, a in enumerate(syms):
            for j, b in enumerate(syms):
                cov = cov_matrix.get((a, b))
                if cov is None and a != b:
                    cov = cov_matrix.get((b, a), 0.0)
                if cov is None:
                    cov = 0.0
                var += allocations[a] * allocations[b] * float(cov)
        if var > limit:
            self.enabled = False
            self.last_kill_reason = "covariance_limit"
            return False
        return True

    # ------------------------------------------------------------------
    # Fill / PnL updates
    def on_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        venue: str | None = None,
        atr: float | None = None,
    ) -> None:
        """Update internal position books after a fill."""
        with self._lock:
            self.add_fill(side, qty, price=price)
            self.guard.update_position_on_order(symbol, side, qty, venue=venue)
            if venue != "paper":
                self.account.update_open_order(symbol, side, -abs(qty))
                delta = qty if side == "buy" else -qty
                self.account.update_position(symbol, delta, price=price)
            if venue is not None:
                book = self.guard.st.venue_positions.get(venue, {})
                venue_book = self.positions_multi.setdefault(venue, {})
                venue_book[symbol] = float(book.get(symbol, 0.0))
            cur_qty, _ = self.account.current_exposure(symbol)
            if abs(cur_qty) < self._min_order_qty:
                self.trades.pop(symbol, None)
                return
            trade = self.trades.get(symbol, {})
            trade.update(
                {
                    "side": "buy" if cur_qty > 0 else "sell",
                    "qty": abs(cur_qty),
                    "entry_price": self._entry_price,
                }
            )
            trade.setdefault("opened_at", time.time())
            if atr is not None:
                trade["atr"] = float(atr)
            entry_price = trade.get("entry_price")
            if entry_price is not None:
                trade.setdefault(
                    "stop",
                    self.initial_stop(entry_price, trade["side"], atr),
                )
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
