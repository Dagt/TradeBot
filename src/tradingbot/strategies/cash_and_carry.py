from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .base import Strategy, Signal, record_signal_metrics

try:  # optional persistence
    from ..storage.timescale import get_engine, insert_cross_signal
    _CAN_PG = True
except Exception:  # pragma: no cover - optional
    _CAN_PG = False

log = logging.getLogger(__name__)

@dataclass
class CashCarryConfig:
    """Configuration parameters for :class:`CashAndCarry`.

    The dataclass is kept for optional persistence features, but the strategy
    itself accepts parameters via ``**kwargs`` and will build this config
    internally.
    """

    symbol: str = ""
    spot_exchange: str = "spot"
    perp_exchange: str = "perp"
    threshold: float = 0.0  # minimum basis to act
    persist_pg: bool = False

class CashAndCarry(Strategy):
    """Simple cash-and-carry strategy using spot vs perpetual futures funding.

    Parameters are supplied via ``**kwargs`` and mapped to
    :class:`CashCarryConfig`.  Only ``threshold`` is relevant for the basic
    strategy logic, while the rest of the fields are used for optional signal
    persistence.

    Parameters
    ----------
    threshold : float, optional
        Minimum basis required to trigger a trade, by default ``0.0``.
    symbol, spot_exchange, perp_exchange, persist_pg : optional
        Passed through to :class:`CashCarryConfig` for persistence.
    """

    name = "cash_and_carry"

    def __init__(self, cfg: CashCarryConfig | None = None, **kwargs):
        """Crear la estrategia.

        Puede pasarse una instancia de :class:`CashCarryConfig` como primer
        argumento o, alternativamente, los parámetros vía ``**kwargs``.
        """

        self.cfg = cfg or CashCarryConfig(**kwargs)
        self.engine = get_engine() if (self.cfg.persist_pg and _CAN_PG) else None

    @record_signal_metrics
    def on_bar(self, bar: dict) -> Optional[Signal]:
        spot = bar.get("spot")
        perp = bar.get("perp")
        funding = bar.get("funding")
        if spot is None or perp is None or funding is None:
            return None
        basis = (perp - spot) / spot
        if funding > 0 and basis > self.cfg.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("buy", basis)
        if funding < 0 and basis < -self.cfg.threshold:
            self._persist_signal(basis, spot, perp)
            return Signal("sell", -basis)
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
