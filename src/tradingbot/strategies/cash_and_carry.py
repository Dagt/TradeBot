from __future__ import annotations

import logging
from typing import Optional

from .base import Strategy, Signal

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_cross_signal
    _CAN_PG = True
except Exception:  # pragma: no cover - optional
    _CAN_PG = False

log = logging.getLogger(__name__)

class CashAndCarry(Strategy):
    """Simple cash-and-carry strategy using spot vs perpetual futures funding.

    Parameters are accepted via ``**kwargs`` to ease dynamic
    configuration.

    Args:
        symbol (str): Trading symbol, e.g. ``"BTCUSDT"``.
        spot_exchange (str): Name of the spot venue. Default ``"spot"``.
        perp_exchange (str): Name of the perpetual venue. Default ``"perp"``.
        threshold (float): Minimum basis to emit a signal. Default ``0``.
        persist_pg (bool): Whether to persist signals to TimescaleDB. Default
            ``False``.
    """

    name = "cash_and_carry"

    def __init__(self, **kwargs):
        self.symbol = kwargs.get("symbol", "")
        self.spot_exchange = kwargs.get("spot_exchange", "spot")
        self.perp_exchange = kwargs.get("perp_exchange", "perp")
        self.threshold = float(kwargs.get("threshold", 0.0))
        self.persist_pg = bool(kwargs.get("persist_pg", False))
        self.engine = get_engine() if (self.persist_pg and _CAN_PG) else None

    def on_bar(self, bar: dict) -> Optional[Signal]:
        spot = bar.get("spot")
        perp = bar.get("perp")
        funding = bar.get("funding")
        if spot is None or perp is None or funding is None:
            return None
        basis = (perp - spot) / spot
        if funding > 0 and basis > self.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("long", basis)
        if funding < 0 and basis < -self.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("short", -basis)
        return Signal("flat", 0.0)

    def _persist_signal(self, basis: float, spot_px: float, perp_px: float) -> None:
        if self.engine is None:
            return
        try:
            insert_cross_signal(
                self.engine,
                symbol=self.symbol,
                spot_exchange=self.spot_exchange,
                perp_exchange=self.perp_exchange,
                spot_px=spot_px,
                perp_px=perp_px,
                edge=basis,
            )
        except Exception as e:  # pragma: no cover - logging only
            log.debug("No se pudo insertar cash&carry signal: %s", e)
