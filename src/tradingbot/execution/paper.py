# src/tradingbot/execution/paper.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, AsyncIterator, AsyncIterable
from datetime import datetime
import csv

from ..adapters.base import ExchangeAdapter
from .order_types import Order
from ..config import settings

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

    def __init__(
        self,
        maker_fee_bps: float | None = None,
        taker_fee_bps: float | None = None,
        fee_bps: float | None = None,
    ):
        self.state = PaperState()
        mf = maker_fee_bps
        if mf is None:
            mf = fee_bps if fee_bps is not None else settings.paper_maker_fee_bps
        self.maker_fee_bps = float(mf)
        default_taker = (
            settings.paper_taker_fee_bps
            if settings.paper_taker_fee_bps is not None
            else self.maker_fee_bps
        )
        self.taker_fee_bps = float(
            taker_fee_bps if taker_fee_bps is not None else default_taker
        )

    def update_last_price(self, symbol: str, px: float):
        self.state.last_px[symbol] = px

    async def stream_trades(
        self,
        symbol: str,
        source: str | Iterable[dict] | AsyncIterable[dict] | None = None,
    ) -> AsyncIterator[dict]:
        """Reproduce trades desde ``source`` y actualiza ``last_px``.

        ``source`` puede ser un path a CSV, una secuencia in-memory
        (lista/generador) o un ``async`` generator. Cada item debe contener
        al menos ``price`` y ``qty``; ``ts`` y ``side`` son opcionales.
        """

        if source is None:
            raise ValueError("source es requerido para PaperAdapter.stream_trades")

        def _prep(trade_dict):
            ts = trade_dict.get("ts")
            price = float(trade_dict["price"])
            qty = float(trade_dict.get("qty", 0.0))
            side = trade_dict.get("side", "buy")
            norm = self.normalize_trade(symbol, ts, price, qty, side)
            self.update_last_price(symbol, price)
            return norm

        if isinstance(source, str):
            with open(source, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    yield _prep(row)
        elif hasattr(source, "__aiter__"):
            async for trade in source:  # type: ignore[operator]
                yield _prep(trade)
        else:
            for trade in source:  # type: ignore[operator]
                yield _prep(trade)

    def _next_order_id(self) -> str:
        self.state.order_id_seq += 1
        return f"paper-{self.state.order_id_seq}"

    def _apply_fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        px: float,
        maker: bool,
        ts: Optional[datetime] = None,
    ) -> dict:
        # fees
        fee_bps = self.maker_fee_bps if maker else self.taker_fee_bps
        fee = abs(qty) * px * (fee_bps / 10000.0)
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
            "fee_usdt": fee,
            "fee_type": "maker" if maker else "taker",
            "fee_bps": fee_bps,
            "pos_qty": pos.qty,
            "pos_avg_px": pos.avg_px,
            "realized_pnl": self.state.realized_pnl,
            "cash": self.state.cash,
            "ts": (ts.isoformat() if ts else None),
        }

    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        post_only: bool = False,
        time_in_force: str | None = None,
        iceberg_qty: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        order_id = self._next_order_id()
        last = self.state.last_px.get(symbol)
        if last is None:
            return {"status": "rejected", "reason": "no_last_price", "order_id": order_id}

        # Determine maker/taker
        maker = post_only or (
            type_.lower() == "limit" and time_in_force not in {"IOC", "FOK"}
        )

        # Estrategia simple: market llena a last; limit solo llena si cruza inmediatamente
        px_exec = None
        if type_.lower() == "market":
            px_exec = last
            maker = False
        elif type_.lower() == "limit":
            if side == "buy" and price is not None and price >= last:
                px_exec = last
            elif side == "sell" and price is not None and price <= last:
                px_exec = last
            else:
                status = "canceled" if time_in_force in {"IOC", "FOK"} else "new"
                return {"status": status, "order_id": order_id, "note": "limit no cruzó; no simulo book"}
        else:
            return {"status": "rejected", "reason": "type_not_supported", "order_id": order_id}

        fill = self._apply_fill(symbol, side, qty, px_exec, maker)
        return {
            "status": "filled",
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": px_exec,
            **fill,
        }

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
