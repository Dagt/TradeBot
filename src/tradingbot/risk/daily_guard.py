# src/tradingbot/risk/daily_guard.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone, date
import logging
from typing import Dict

from ..utils.metrics import RISK_EVENTS
from ..storage import timescale

log = logging.getLogger(__name__)

@dataclass
class GuardLimits:
    daily_max_loss_usdt: float = 100.0           # pérdida neta diaria permitida
    daily_max_drawdown_pct: float = 0.05         # dd% relativo a equity pico intradía (0.05 = 5%)
    max_consecutive_losses: int = 3              # SLs o pérdidas consecutivas
    halt_action: str = "close_all"               # "close_all" | "pause_only"

class DailyGuard:
    """
    Controla pérdidas diarias, drawdown y racha de pérdidas.
    Mantiene equity intradía, pérdidas consecutivas y fecha de corte (UTC).
    """
    def __init__(self, limits: GuardLimits, venue: str, storage_engine=None):
        self.lim = limits
        self.venue = venue
        self._cur_day: date | None = None
        self._equity_open: float = 0.0
        self._equity_peak: float = 0.0
        self._equity_last: float = 0.0
        self._realized_today: float = 0.0
        self._consec_losses: int = 0
        self._halted: bool = False
        self._engine = storage_engine

    def reset_for_new_day(self, now: datetime, equity_now: float):
        self._cur_day = now.date()
        self._equity_open = equity_now
        self._equity_peak = equity_now
        self._equity_last = equity_now
        self._realized_today = 0.0
        self._consec_losses = 0
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
        self._realized_today += float(delta_rpnl)
        if delta_rpnl < 0:
            self._consec_losses += 1
        elif delta_rpnl > 0:
            self._consec_losses = 0

    def check_halt(self) -> tuple[bool, str]:
        if self._halted:
            return True, "already_halted"

        # Regla 1: pérdida neta diaria
        if self._realized_today <= -abs(self.lim.daily_max_loss_usdt):
            self._halted = True
            RISK_EVENTS.labels(event_type="daily_max_loss").inc()
            if self._engine is not None:
                try:
                    timescale.insert_risk_event(
                        self._engine,
                        venue=self.venue,
                        symbol="",
                        kind="daily_max_loss",
                        message="",
                    )
                except Exception:
                    pass
            return True, "daily_max_loss"

        # Regla 2: drawdown desde equity_peak
        if self._equity_peak > 0:
            dd = (self._equity_peak - self._equity_last) / self._equity_peak
            if dd >= self.lim.daily_max_drawdown_pct:
                self._halted = True
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

        # Regla 3: racha de pérdidas
        if self._consec_losses >= self.lim.max_consecutive_losses:
            self._halted = True
            RISK_EVENTS.labels(event_type="consecutive_losses").inc()
            if self._engine is not None:
                try:
                    timescale.insert_risk_event(
                        self._engine,
                        venue=self.venue,
                        symbol="",
                        kind="consecutive_losses",
                        message="",
                    )
                except Exception:
                    pass
            return True, "consecutive_losses"

        return False, ""

