# src/tradingbot/execution/paper.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

from ..adapters.base import ExchangeAdapter
from .order_types import Order

log = logging.getLogger(__name__)

@dataclass
class PaperPosition:
    qty: float = 0.0
    avg_px: float = 0.0

@dataclass
class PaperState:
    last_px: Dict[str, float] = field(default_factory=dict)      # symbol -> last price
    pos: Dict[str, PaperPosition] = field(default_factory=dict)  # symbol -> position
    cash: float = 0.0
    realized_pnl: float = 0.0
    order_id_seq: int = 0

class PaperAdapter(ExchangeAdapter):
    """
    Adapter que implementa ExchangeAdapter pero en papel.
    Ejecuta órdenes al precio last actual (market) o mejor (limit si cruza).
    """
    name = "paper"

    def __init__(self, fee_bps: float = 0.0):
        self.state = PaperState()
        self.fee_bps = fee_bps

    def update_last_price(self, symbol: str, px: float):
        self.state.last_px[symbol] = px

    async def stream_trades(self, symbol: str):
        # PaperAdapter no produce stream: úsalo con un adapter WS externo y llama update_last_price()
        raise NotImplementedError("PaperAdapter no genera stream; úsalo como broker simulado")

    def _next_order_id(self) -> str:
        self.state.order_id_seq += 1
        return f"paper-{self.state.order_id_seq}"

    def _apply_fill(self, symbol: str, side: str, qty: float, px: float, ts: Optional[datetime] = None) -> dict:
        # fees
        fee = abs(qty) * px * (self.fee_bps / 10000.0)
        # cash/pnl y posición
        pos = self.state.pos.setdefault(symbol, PaperPosition())
        if side == "buy":
            new_qty = pos.qty + qty
            if new_qty == 0:
                pos.avg_px = 0.0
            elif pos.qty >= 0:
                # promedia si seguimos largos
                pos.avg_px = (pos.avg_px * pos.qty + qty * px) / new_qty
            else:
                # estábamos cortos, realizar PnL por cierre parcial
                closed = min(qty, -pos.qty)
                self.state.realized_pnl += (pos.avg_px - px) * closed
            pos.qty = new_qty
            self.state.cash -= qty * px + fee
        elif side == "sell":
            new_qty = pos.qty - qty
            if new_qty == 0:
                pos.avg_px = 0.0
            elif pos.qty <= 0:
                pos.avg_px = (pos.avg_px * (-pos.qty) + qty * px) / (-new_qty)
            else:
                closed = min(qty, pos.qty)
                self.state.realized_pnl += (px - pos.avg_px) * closed
            pos.qty = new_qty
            self.state.cash += qty * px - fee
        else:
            raise ValueError("side debe ser buy/sell")

        return {
            "filled": True,
            "fee": fee,
            "pos_qty": pos.qty,
            "pos_avg_px": pos.avg_px,
            "realized_pnl": self.state.realized_pnl,
            "cash": self.state.cash,
            "ts": (ts.isoformat() if ts else None),
        }

    async def place_order(self, symbol: str, side: str, type_: str, qty: float, price: float | None = None) -> dict:
        order_id = self._next_order_id()
        last = self.state.last_px.get(symbol)
        if last is None:
            return {"status": "rejected", "reason": "no_last_price", "order_id": order_id}

        # Estrategia simple: market llena a last; limit solo llena si cruza inmediatamente
        px_exec = None
        if type_.lower() == "market":
            px_exec = last
        elif type_.lower() == "limit":
            if side == "buy" and price is not None and price >= last:
                px_exec = last
            elif side == "sell" and price is not None and price <= last:
                px_exec = last
            else:
                return {"status": "new", "order_id": order_id, "note": "limit no cruzó; no simulo book"}
        else:
            return {"status": "rejected", "reason": "type_not_supported", "order_id": order_id}

        fill = self._apply_fill(symbol, side, qty, px_exec)
        return {"status": "filled", "order_id": order_id, "symbol": symbol, "side": side, "qty": qty, "price": px_exec, **fill}

    async def cancel_order(self, order_id: str) -> dict:
        # Este simulador no mantiene colas pendientes (limit no cruzado = no entra al libro)
        return {"status": "canceled", "order_id": order_id}

    # Helpers de estado para inspección
    def equity(self, mark_prices: dict[str, float] | None = None) -> float:
        eq = self.state.cash + self.state.realized_pnl
        marks = mark_prices or self.state.last_px
        for sym, p in self.state.pos.items():
            mark = marks.get(sym)
            if mark is not None:
                eq += p.qty * mark
        return eq
