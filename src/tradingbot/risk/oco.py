# src/tradingbot/risk/oco.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal, Callable
from datetime import datetime, timezone

from sqlalchemy import text

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
        self._tiers_state: Dict[str, dict] = {}

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
                       close_fn, trail_pct: Optional[float] = None,
                       tp_tiers: Optional[list[tuple[float, float]]] = None):
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
        # === TP parciales por tiers (opcional) ===
        # tp_tiers es lista de (r_multiple, fraction), ej: [(1.0, 0.5), (2.0, 0.5)]
        # R = distancia de riesgo = |entry - sl_price|
        if tp_tiers and len(tp_tiers) > 0:
            # inicializar estado si no existe
            st = self._tiers_state.get(symbol)
            if st is None or st.get("orig_qty", 0.0) <= 0 or st.get("tiers") != tp_tiers:
                self._tiers_state[symbol] = {
                    "orig_qty": abs(pos_qty),
                    "done": [False] * len(tp_tiers),
                    "tiers": list(tp_tiers),
                }
                st = self._tiers_state[symbol]

            # Recalcular targets por si sl_price cambió por trailing (R cambia)
            R = abs(oco.entry_price - oco.sl_price)
            if R > 0:
                # Revisa cada tier en orden
                for i, (r_mult, frac) in enumerate(st["tiers"]):
                    if st["done"][i]:
                        continue
                    # precio objetivo según lado
                    if oco.side == "long":
                        tgt = oco.entry_price + r_mult * R
                        hit = px >= tgt
                    else:
                        tgt = oco.entry_price - r_mult * R
                        hit = px <= tgt

                    if hit:
                        # qty a cerrar = fracción de la cantidad ORIGINAL
                        qty_to_close = max(0.0, float(frac) * float(st["orig_qty"]))
                        if qty_to_close > 0:
                            ok = await close_fn(symbol, f"TAKE_PROFIT_TIER_{i+1}", qty_to_close, px)
                            if ok:
                                # marca tier como realizado
                                st["done"][i] = True
                                # reduce la qty restante del OCO (persistimos con on_update)
                                oco.qty = max(0.0, float(oco.qty) - float(qty_to_close))
                                if self._on_update:
                                    self._on_update(symbol, oco)

                                # si ya no queda qty, marcamos TP total
                                if oco.qty <= 1e-12:
                                    oco.triggered, oco.triggered_at = "TP", datetime.now(timezone.utc)
                                    if self._on_trigger:
                                        self._on_trigger(symbol, "TP")
                                    return
                        # seguimos para permitir múltiples tiers en el mismo tick
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


# ---------------------------------------------------------------------------
# Persistence helpers
def load_open_oco(engine, venue: str, symbols: list[str]) -> Dict[str, OcoOrder]:
    """Load active OCO orders for symbols from database."""

    if not symbols:
        return {}
    placeholders = ",".join(f":s{i}" for i in range(len(symbols)))
    params = {"venue": venue}
    for i, sym in enumerate(symbols):
        params[f"s{i}"] = sym
    sql = text(
        f"""
        SELECT symbol, side, qty, entry_price, sl_price, tp_price
        FROM market.oco_orders
        WHERE venue=:venue AND status='active' AND symbol IN ({placeholders})
        """
    )
    out: Dict[str, OcoOrder] = {}
    with engine.begin() as conn:
        for row in conn.execute(sql, params).mappings():
            out[row["symbol"]] = OcoOrder(
                symbol=row["symbol"],
                side=row["side"],
                qty=float(row["qty"]),
                entry_price=float(row["entry_price"]),
                sl_price=float(row["sl_price"]),
                tp_price=float(row["tp_price"]),
                best_price_since_entry=float(row["entry_price"]),
            )
    return out
