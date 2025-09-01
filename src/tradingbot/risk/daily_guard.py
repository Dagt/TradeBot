# src/tradingbot/risk/daily_guard.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, date
import logging
import asyncio

from ..utils.metrics import RISK_EVENTS
from ..storage import timescale
from ..config import settings

log = logging.getLogger(__name__)

@dataclass
class GuardLimits:
    daily_max_loss_pct: float = 0.05            # pérdida diaria permitida (% equity)
    daily_max_drawdown_pct: float = 0.05        # dd% relativo a equity pico intradía
    halt_action: str = "close_all"              # "close_all" | "pause_only"

class DailyGuard:
    """
    Controla pérdidas diarias y drawdown intradía.
    Mantiene equity intradía y fecha de corte (UTC).
    """
    def __init__(self, limits: GuardLimits, venue: str, storage_engine=None):
        self.lim = limits
        self.venue = venue
        self._cur_day: date | None = None
        self._equity_open: float = 0.0
        self._equity_peak: float = 0.0
        self._equity_last: float = 0.0
        self._realized_today: float = 0.0
        self._halted: bool = False
        self._engine = storage_engine

    def reset_for_new_day(self, now: datetime, equity_now: float):
        self._cur_day = now.date()
        self._equity_open = equity_now
        self._equity_peak = equity_now
        self._equity_last = equity_now
        self._realized_today = 0.0
        self._halted = False
        log.info("[DG] Nueva sesión %s: equity_open=%.2f", self._cur_day, equity_now)

    @property
    def halted(self) -> bool:
        return self._halted

    def on_mark(self, now: datetime, equity_now: float):
        # Rollover de día (UTC)
        if self._cur_day is None or now.date() != self._cur_day:
            self.reset_for_new_day(now, equity_now)
            return
        self._equity_last = equity_now
        if equity_now > self._equity_peak:
            self._equity_peak = equity_now

    def on_realized_delta(self, delta_rpnl: float):
        """Track realized PnL delta (for logging/metrics)."""
        self._realized_today += float(delta_rpnl)

    def apply_halt_action(self, broker) -> None:
        """Apply configured halt action through ``broker``.

        If ``halt_action`` is ``"close_all"`` market orders are sent
        asynchronously to close every open position known by ``broker``.
        """
        if broker is None or self.lim.halt_action != "close_all":
            return
        try:
            pos_book = getattr(getattr(broker, "state", object()), "pos", {})
            place = getattr(broker, "place_order", None)
            if not pos_book or place is None:
                return
            tick = getattr(settings, "tick_size", 0.0)
            for sym, p in list(pos_book.items()):
                qty = getattr(p, "qty", 0.0)
                if abs(qty) <= 0:
                    continue
                side = "sell" if qty > 0 else "buy"
                last_px = getattr(getattr(broker, "state", object()), "last_px", {}).get(sym)
                if last_px is None:
                    continue
                price = last_px - tick if side == "sell" else last_px + tick
                place_limit = getattr(broker, "place_limit", None)
                if place_limit is None:
                    continue
                asyncio.create_task(place_limit(sym, side, price, abs(qty), tif="IOC"))
        except Exception:  # pragma: no cover - safety guard
            log.exception("[DG] apply_halt_action failed")

    def check_halt(self, broker=None) -> tuple[bool, str]:
        if self._halted:
            return True, "already_halted"

        # Regla 1: pérdida diaria relativa a equity de apertura
        if self._equity_open > 0:
            loss_pct = (self._equity_last - self._equity_open) / self._equity_open
            if loss_pct <= -abs(self.lim.daily_max_loss_pct):
                self._halted = True
                self.apply_halt_action(broker)
                RISK_EVENTS.labels(event_type="daily_loss").inc()
                if self._engine is not None:
                    try:
                        timescale.insert_risk_event(
                            self._engine,
                            venue=self.venue,
                            symbol="",
                            kind="daily_loss",
                            message="",
                        )
                    except Exception:
                        pass
                return True, "daily_loss"

        # Regla 2: drawdown desde equity_peak
        if self._equity_peak > 0:
            dd = (self._equity_peak - self._equity_last) / self._equity_peak
            if dd >= self.lim.daily_max_drawdown_pct:
                self._halted = True
                self.apply_halt_action(broker)
                RISK_EVENTS.labels(event_type="daily_drawdown").inc()
                if self._engine is not None:
                    try:
                        timescale.insert_risk_event(
                            self._engine,
                            venue=self.venue,
                            symbol="",
                            kind="daily_drawdown",
                            message="",
                        )
                    except Exception:
                        pass
                return True, "daily_drawdown"

        return False, ""
