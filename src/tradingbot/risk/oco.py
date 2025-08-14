# src/tradingbot/risk/oco.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Callable
from datetime import datetime, timezone

Side = Literal["long", "short"]

@dataclass
class OcoOrder:
    symbol: str
    side: Side
    qty: float
    entry_price: float
    sl_price: float
    tp_price: float
    # extremos a favor para trailing
    best_price_since_entry: Optional[float] = None  # high si long, low si short
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered: Optional[str] = None
    triggered_at: Optional[datetime] = None

class OcoBook:
    """
    Libro OCO simple en memoria con callbacks opcionales:
      on_create(symbol, order), on_update(symbol, order), on_cancel(symbol),
      on_trigger(symbol, which) where which in {"SL","TP"}.
    """
    def __init__(self,
                 on_create: Callable[[str, OcoOrder], None] | None = None,
                 on_update: Callable[[str, OcoOrder], None] | None = None,
                 on_cancel: Callable[[str], None] | None = None,
                 on_trigger: Callable[[str, str], None] | None = None):
        self._items: Dict[str, OcoOrder] = {}
        self._on_create = on_create
        self._on_update = on_update
        self._on_cancel = on_cancel
        self._on_trigger = on_trigger

    def preload(self, items: Dict[str, OcoOrder]):
        self._items.update(items)

    def get(self, symbol: str) -> Optional[OcoOrder]:
        return self._items.get(symbol)

    def cancel(self, symbol: str):
        if symbol in self._items:
            self._items.pop(symbol, None)
            if self._on_cancel:
                self._on_cancel(symbol)

    def create_or_update(self, *, symbol: str, side: Side, qty: float, entry_price: float,
                         sl_pct: float, tp_pct: float):
        qty = max(0.0, float(qty))
        if qty <= 0 or entry_price <= 0:
            self.cancel(symbol)
            return
        if side == "long":
            sl_price = entry_price * (1.0 - abs(sl_pct))
            tp_price = entry_price * (1.0 + abs(tp_pct))
        else:
            sl_price = entry_price * (1.0 + abs(sl_pct))
            tp_price = entry_price * (1.0 - abs(tp_pct))

        prev = self._items.get(symbol)
        if prev:
            prev.qty = qty
            prev.entry_price = entry_price
            prev.sl_price = sl_price
            prev.tp_price = tp_price
            # reinicia extremo a favor al nuevo entry
            prev.best_price_since_entry = entry_price
            prev.triggered = None
            prev.triggered_at = None
            if self._on_update:
                self._on_update(symbol, prev)
        else:
            order = OcoOrder(symbol=symbol, side=side, qty=qty,
                             entry_price=entry_price, sl_price=sl_price, tp_price=tp_price)
            # inicializa extremo a favor
            order.best_price_since_entry = entry_price
            self._items[symbol] = order
            if self._on_create:
                self._on_create(symbol, order)

    async def evaluate(self, symbol: str, last_price: float,
                       pos_qty: float, entry_price: float,
                       sl_pct: float, tp_pct: float,
                       close_fn, trail_pct: Optional[float] = None):
        if abs(pos_qty) <= 0 or entry_price <= 0:
            self.cancel(symbol)
            return

        side: Side = "long" if pos_qty > 0 else "short"
        oco = self.get(symbol)
        if not oco or oco.side != side:
            self.create_or_update(symbol=symbol, side=side, qty=abs(pos_qty),
                                  entry_price=entry_price, sl_pct=sl_pct, tp_pct=tp_pct)
            oco = self.get(symbol)

        if oco and (abs(oco.qty - abs(pos_qty)) > 1e-12 or abs(oco.entry_price - entry_price) > 1e-12):
            self.create_or_update(symbol=symbol, side=side, qty=abs(pos_qty),
                                  entry_price=entry_price, sl_pct=sl_pct, tp_pct=tp_pct)
            oco = self.get(symbol)

        if not oco or oco.triggered:
            return

        # >>> Trailing stop: actualiza sl_price a favor si corresponde
        self._update_trailing(oco, last_price, trail_pct)
        px = float(last_price)

        if (side == "long" and px <= oco.sl_price) or (side == "short" and px >= oco.sl_price):
            ok = await close_fn(symbol, "STOP_LOSS", min(oco.qty, abs(pos_qty)), px)
            if ok:
                oco.triggered, oco.triggered_at = "SL", datetime.now(timezone.utc)
                if self._on_trigger:
                    self._on_trigger(symbol, "SL")
            return

        if (side == "long" and px >= oco.tp_price) or (side == "short" and px <= oco.tp_price):
            ok = await close_fn(symbol, "TAKE_PROFIT", min(oco.qty, abs(pos_qty)), px)
            if ok:
                oco.triggered, oco.triggered_at = "TP", datetime.now(timezone.utc)
                if self._on_trigger:
                    self._on_trigger(symbol, "TP")
            return

    def _update_trailing(self, oco: OcoOrder, last_price: float, trail_pct: Optional[float]):
        """Ajusta sl_price a favor, manteniendo distancia trail_pct desde el mejor precio."""
        if not trail_pct or trail_pct <= 0:
            return

        px = float(last_price)
        if oco.side == "long":
            # registrar nuevo máximo a favor
            if oco.best_price_since_entry is None or px > oco.best_price_since_entry:
                oco.best_price_since_entry = px
            # nuevo SL propuesto: max - trail%
            target_sl = oco.best_price_since_entry * (1.0 - abs(trail_pct))
            # nunca aflojar SL: solo subirlo
            if target_sl > oco.sl_price:
                oco.sl_price = target_sl
                if self._on_update:
                    self._on_update(oco.symbol, oco)
        else:
            # short: registrar nuevo mínimo a favor
            if oco.best_price_since_entry is None or px < oco.best_price_since_entry:
                oco.best_price_since_entry = px
            # nuevo SL propuesto: min + trail%
            target_sl = oco.best_price_since_entry * (1.0 + abs(trail_pct))
            # nunca aflojar SL: solo bajarlo (más cerca del precio en short)
            if target_sl < oco.sl_price:
                oco.sl_price = target_sl
                if self._on_update:
                    self._on_update(oco.symbol, oco)
