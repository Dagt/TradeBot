from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .base import Strategy, Signal

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_cross_signal
    _CAN_PG = True
except Exception:  # pragma: no cover - optional
    _CAN_PG = False

log = logging.getLogger(__name__)

@dataclass
class CashCarryConfig:
    symbol: str
    spot_exchange: str = "spot"
    perp_exchange: str = "perp"
    threshold: float = 0.0  # minimum basis to act
    persist_pg: bool = False

class CashAndCarry(Strategy):
    """Simple cash-and-carry strategy using spot vs perpetual futures funding.

    The strategy looks at the price basis between spot and perpetual markets
    together with the current funding rate.  When funding is positive and the
    perp trades at a premium greater than ``threshold`` the strategy issues a
    ``long`` signal (long spot/short perp).  When funding is negative and the
    perp trades at a discount beyond the threshold a ``short`` signal is
    produced (short spot/long perp).
    """

    name = "cash_and_carry"

    def __init__(self, cfg: CashCarryConfig):
        self.cfg = cfg
        self.engine = get_engine() if (cfg.persist_pg and _CAN_PG) else None

    def on_bar(self, bar: dict) -> Optional[Signal]:
        spot = bar.get("spot")
        perp = bar.get("perp")
        funding = bar.get("funding")
        if spot is None or perp is None or funding is None:
            return None
        basis = (perp - spot) / spot
        if funding > 0 and basis > self.cfg.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("long", basis)
        if funding < 0 and basis < -self.cfg.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("short", -basis)
        return Signal("flat", 0.0)

    def _persist_signal(self, basis: float, spot_px: float, perp_px: float) -> None:
        if self.engine is None:
            return
        try:
            insert_cross_signal(
                self.engine,
                symbol=self.cfg.symbol,
                spot_exchange=self.cfg.spot_exchange,
                perp_exchange=self.cfg.perp_exchange,
                spot_px=spot_px,
                perp_px=perp_px,
                edge=basis,
            )
        except Exception as e:  # pragma: no cover - logging only
            log.debug("No se pudo insertar cash&carry signal: %s", e)
