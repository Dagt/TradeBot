from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..bus import EventBus


@dataclass
class RiskLimits:
    """Configuración de límites básicos de riesgo."""

    max_notional: float = 0.0
    max_concurrent_orders: int = 0
    daily_dd_limit: float = 0.0
    hard_pnl_stop: float = 0.0


class LimitTracker:
    """Estado y chequeos asociados a :class:`RiskLimits`."""

    def __init__(self, limits: RiskLimits, bus: EventBus | None = None) -> None:
        self.lim = limits
        self.bus = bus
        self.open_orders: int = 0
        self.daily_pnl: float = 0.0
        self._daily_peak: float = 0.0
        self.blocked: bool = False

    # ------------------------------------------------------------------
    # Order helpers
    def register_order(self, notional: float) -> bool:
        """Registrar una nueva orden si no viola límites."""
        if self.blocked:
            return False
        if self.lim.max_notional > 0 and notional > self.lim.max_notional:
            return False
        if (
            self.lim.max_concurrent_orders > 0
            and self.open_orders >= self.lim.max_concurrent_orders
        ):
            return False
        self.open_orders += 1
        return True

    def complete_order(self) -> None:
        if self.open_orders > 0:
            self.open_orders -= 1

    # ------------------------------------------------------------------
    # PnL / drawdown tracking
    def update_pnl(self, delta: float) -> None:
        if self.blocked:
            return
        self.daily_pnl += float(delta)
        if self.daily_pnl > self._daily_peak:
            self._daily_peak = self.daily_pnl
        drawdown = self._daily_peak - self.daily_pnl
        if self.lim.daily_dd_limit > 0 and drawdown >= self.lim.daily_dd_limit:
            self.blocked = True
            if self.bus is not None:
                asyncio.create_task(
                    self.bus.publish("risk:blocked", {"reason": "daily_dd_limit"})
                )
        if (
            self.lim.hard_pnl_stop > 0
            and self.daily_pnl <= -abs(self.lim.hard_pnl_stop)
        ):
            self.blocked = True
            if self.bus is not None:
                asyncio.create_task(
                    self.bus.publish("risk:blocked", {"reason": "hard_pnl_stop"})
                )

    def reset(self) -> None:
        self.open_orders = 0
        self.daily_pnl = 0.0
        self._daily_peak = 0.0
        self.blocked = False
