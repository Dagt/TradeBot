# src/tradingbot/execution/paper.py
from __future__ import annotations
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable, AsyncIterator, AsyncIterable, Mapping, Any
from datetime import datetime
import csv

from ..adapters.base import ExchangeAdapter
from .order_types import Order
from ..config import settings
from .order_sizer import adjust_qty
from .slippage import impact_by_depth
try:  # Optional import for environments without backtesting module
    from ..backtesting.engine import SlippageModel  # type: ignore
except Exception:  # pragma: no cover - fallback when backtesting is unavailable
    SlippageModel = None  # type: ignore

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
    trade_id_seq: int = 0
    book: Dict[str, Dict[str, "BookOrder"]] = field(default_factory=dict)

@dataclass
class BookOrder:
    order_id: str
    symbol: str
    side: str
    price: float
    qty: float
    pending_qty: float
    placed_at: datetime
    timeout: float | None = None


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
        latency: float = 0.0,
        *,
        min_notional: float = 0.0,
        step_size: float = 0.0,
        slip_bps_per_qty: float = 0.0,
        slippage_model: "SlippageModel" | None = None,
    ):
        super().__init__()
        self.state = PaperState()
        mf = (
            maker_fee_bps
            if maker_fee_bps is not None
            else (fee_bps if fee_bps is not None else settings.paper_maker_fee_bps)
        )
        self.maker_fee_bps = float(mf)
        self.taker_fee_bps = float(
            taker_fee_bps
            if taker_fee_bps is not None
            else settings.paper_taker_fee_bps
        )
        self.latency = float(latency)
        # Common limits used to mirror exchange behaviour during sizing.
        self.min_notional = float(min_notional)
        self.step_size = float(step_size)
        self.slip_bps_per_qty = float(slip_bps_per_qty)
        self.slippage_model = slippage_model

    def update_last_price(
        self,
        symbol: str,
        px: float,
        qty: float | None = None,
        book: Mapping[str, Any] | None = None,
    ):
        self.state.last_px[symbol] = px
        self.account.mark_price(symbol, px)
        return self._match_book(symbol, px, qty, book)

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
            self.update_last_price(symbol, price, qty)
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

    def _next_trade_id(self) -> str:
        self.state.trade_id_seq += 1
        return f"trade-{self.state.trade_id_seq}"

    def _match_book(
        self,
        symbol: str,
        px: float,
        qty: float | None = None,
        book: Mapping[str, Any] | None = None,
    ) -> list[dict]:
        fills: list[dict] = []
        orders = self.state.book.get(symbol)
        if not orders:
            return fills
        remaining = float("inf") if qty is None else float(qty)
        now = datetime.utcnow()
        # Available top-of-book liquidity if provided
        if book is not None:
            remaining_asks = float(book.get("ask_size", float("inf")))
            remaining_bids = float(book.get("bid_size", float("inf")))
        else:
            remaining_asks = remaining_bids = float("inf")
        for oid in list(orders.keys()):
            if remaining <= 0:
                break
            order = orders[oid]
            crosses = (
                (order.side == "buy" and px <= order.price)
                or (order.side == "sell" and px >= order.price)
            )
            if not crosses:
                continue
            fee_bps = self.maker_fee_bps
            pos = self.state.pos.get(symbol, PaperPosition())
            if order.side == "sell":
                avail = pos.qty
                book_avail = remaining_bids
            else:
                avail = (
                    self.state.cash / (px * (1 + fee_bps / 10000.0))
                    if self.state.cash > 0
                    else 0.0
                )
                book_avail = remaining_asks
            fill_qty = min(order.pending_qty, remaining, avail, book_avail)
            if fill_qty <= 0:
                continue
            px_slip, slip_bps = self._apply_slippage(order.side, fill_qty, px, book)
            fill = self._apply_fill(symbol, order.side, fill_qty, px_slip, True)
            order.pending_qty -= fill_qty
            remaining -= fill_qty
            if order.side == "sell":
                remaining_bids -= fill_qty
            else:
                remaining_asks -= fill_qty
            status = "filled" if order.pending_qty <= 0 else "partial"
            time_in_book = (datetime.utcnow() - order.placed_at).total_seconds()
            res = {
                "status": status,
                "order_id": order.order_id,
                "trade_id": self._next_trade_id(),
                "symbol": symbol,
                "side": order.side,
                "qty": fill_qty,
                "filled_qty": fill_qty,
                "price": px_slip,
                "pending_qty": max(order.pending_qty, 0.0),
                "time_in_book": time_in_book,
                "slippage_bps": slip_bps,
                **fill,
            }
            fills.append(res)
            if order.pending_qty <= 0:
                del orders[oid]
        # handle expirations
        for oid, order in list(orders.items()):
            if order.timeout is not None and (now - order.placed_at).total_seconds() >= order.timeout:
                filled_qty = order.qty - order.pending_qty
                status = "partial" if filled_qty > 0 else "expired"
                res = {
                    "status": status,
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": order.side,
                    "price": order.price,
                    "qty": filled_qty,
                    "filled_qty": filled_qty,
                    "pending_qty": order.pending_qty,
                    "time_in_book": (now - order.placed_at).total_seconds(),
                }
                fills.append(res)
                del orders[oid]
        if not orders:
            self.state.book.pop(symbol, None)
        return fills

    def _apply_slippage(
        self,
        side: str,
        qty: float,
        px: float,
        book: Mapping[str, Any] | None = None,
    ) -> tuple[float, float]:
        """Return adjusted price and slippage in basis points.

        Priority is given to an injected :class:`SlippageModel`.  If no model is
        provided, but order book levels are available via ``book`` (``bids`` or
        ``asks`` lists), slippage is estimated by consuming depth using
        :func:`impact_by_depth`.  Finally, a simple linear model based on
        ``slip_bps_per_qty`` acts as a fallback for backwards compatibility.
        """

        # Slippage model path ----------------------------------------------
        if self.slippage_model is not None and book is not None:
            adj = self.slippage_model.adjust(side, qty, px, book)
            if side == "buy":
                slip_bps = (adj - px) / px * 10000.0
            else:
                slip_bps = (px - adj) / px * 10000.0
            return adj, slip_bps

        # Depth based path --------------------------------------------------
        if book is not None:
            levels = None
            if side == "buy" and "asks" in book:
                levels = book["asks"]  # type: ignore[index]
            elif side == "sell" and "bids" in book:
                levels = book["bids"]  # type: ignore[index]
            if levels:
                slip_bps = impact_by_depth(side, qty, levels) or 0.0
                px_slipped = px * (1 + slip_bps / 10000.0)
                return px_slipped, slip_bps

        # Linear fallback ---------------------------------------------------
        slip_bps = self.slip_bps_per_qty * qty
        direction = 1 if side == "buy" else -1
        px_slipped = px * (1 + direction * slip_bps / 10000.0)
        return px_slipped, slip_bps

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
        prev_qty = pos.qty
        if side == "buy":
            new_qty = prev_qty + qty
            if prev_qty < 0:
                closed = min(qty, -prev_qty)
                self.state.realized_pnl += (pos.avg_px - px) * closed
            pos.qty = new_qty
            if new_qty == 0:
                pos.avg_px = 0.0
            elif prev_qty >= 0:
                pos.avg_px = (pos.avg_px * prev_qty + qty * px) / new_qty
            elif new_qty > 0:
                pos.avg_px = px
            self.state.cash -= qty * px + fee
            self.account.update_position(symbol, qty, px)
            self.account.update_cash(-(qty * px + fee))
        elif side == "sell":
            new_qty = prev_qty - qty
            if prev_qty > 0:
                closed = min(qty, prev_qty)
                self.state.realized_pnl += (px - pos.avg_px) * closed
            pos.qty = new_qty
            if new_qty == 0:
                pos.avg_px = 0.0
            elif prev_qty <= 0:
                pos.avg_px = (pos.avg_px * (-prev_qty) + qty * px) / (-new_qty)
            elif new_qty < 0:
                pos.avg_px = px
            self.state.cash += qty * px - fee
            self.account.update_position(symbol, -qty, px)
            self.account.update_cash(qty * px - fee)
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
        timeout: float | None = None,
        iceberg_qty: float | None = None,
        reduce_only: bool = False,
        book: Mapping[str, Any] | None = None,
    ) -> dict:
        order_id = self._next_order_id()
        trade_id = self._next_trade_id()
        last = self.state.last_px.get(symbol)
        if last is None:
            return {
                "status": "rejected",
                "reason": "no_last_price",
                "order_id": order_id,
                "trade_id": trade_id,
            }
        working_px = price if price is not None else last
        qty = adjust_qty(qty, working_px, self.min_notional, self.step_size)
        if qty <= 0:
            return {
                "status": "rejected",
                "reason": "min_notional",
                "order_id": order_id,
                "trade_id": trade_id,
            }

        maker = post_only or (
            type_.lower() == "limit" and time_in_force not in {"IOC", "FOK"}
        )

        px_exec = None
        if type_.lower() == "market":
            px_exec = last
            maker = False
        elif type_.lower() == "limit":
            if side == "buy" and price is not None and price >= last:
                px_exec = last
                maker = False
            elif side == "sell" and price is not None and price <= last:
                px_exec = last
                maker = False
        else:
            return {
                "status": "rejected",
                "reason": "type_not_supported",
                "order_id": order_id,
                "trade_id": trade_id,
            }

        start = time.monotonic()
        if px_exec is None and type_.lower() == "limit" and time_in_force not in {"IOC", "FOK"}:
            fee_bps = self.maker_fee_bps
            pos = self.state.pos.get(symbol, PaperPosition())
            if side == "sell":
                qty = min(qty, pos.qty)
                if qty <= 0:
                    return {
                        "status": "rejected",
                        "reason": "insufficient_position",
                        "order_id": order_id,
                        "trade_id": trade_id,
                    }
            else:
                max_aff = (
                    self.state.cash / (price * (1 + fee_bps / 10000.0))
                    if self.state.cash > 0
                    else 0.0
                )
                qty = min(qty, max_aff)
                if qty <= 0:
                    return {
                        "status": "rejected",
                        "reason": "insufficient_cash",
                        "order_id": order_id,
                        "trade_id": trade_id,
                    }
            order = BookOrder(order_id, symbol, side, float(price), qty, qty, datetime.utcnow(), timeout)
            self.state.book.setdefault(symbol, {})[order_id] = order
            if self.latency:
                await asyncio.sleep(self.latency)
            latency = time.monotonic() - start
            return {
                "status": "new",
                "order_id": order_id,
                "trade_id": trade_id,
                "symbol": symbol,
                "side": side,
                "price": price,
                "qty": 0.0,
                "filled_qty": 0.0,
                "pending_qty": qty,
                "latency": latency,
            }

        if px_exec is None:
            status = "canceled" if time_in_force in {"IOC", "FOK"} else "new"
            if self.latency:
                await asyncio.sleep(self.latency)
            latency = time.monotonic() - start
            return {
                "status": status,
                "order_id": order_id,
                "trade_id": trade_id,
                "latency": latency,
            }

        fee_bps = self.maker_fee_bps if maker else self.taker_fee_bps
        pos = self.state.pos.get(symbol, PaperPosition())
        if side == "sell":
            qty = min(qty, pos.qty)
            book_avail = float(book.get("bid_size", float("inf"))) if book else float("inf")
            qty = min(qty, book_avail)
            if qty <= 0:
                return {
                    "status": "rejected",
                    "reason": "insufficient_position",
                    "order_id": order_id,
                    "trade_id": trade_id,
                }
        else:
            max_aff = (
                self.state.cash / (px_exec * (1 + fee_bps / 10000.0))
                if self.state.cash > 0
                else 0.0
            )
            qty = min(qty, max_aff)
            book_avail = float(book.get("ask_size", float("inf"))) if book else float("inf")
            qty = min(qty, book_avail)
            if qty <= 0:
                return {
                    "status": "rejected",
                    "reason": "insufficient_cash",
                    "order_id": order_id,
                    "trade_id": trade_id,
                }
        px_exec, slip_bps = self._apply_slippage(side, qty, px_exec, book)
        fill = self._apply_fill(symbol, side, qty, px_exec, maker)
        if self.latency:
            await asyncio.sleep(self.latency)
        latency = time.monotonic() - start
        return {
            "status": "filled",
            "order_id": order_id,
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "filled_qty": qty,
            "pending_qty": 0.0,
            "price": px_exec,
            "latency": latency,
            "slippage_bps": slip_bps,
            **fill,
        }

    async def cancel_order(self, order_id: str, symbol: str | None = None) -> dict:
        start = time.monotonic()
        book = self.state.book
        order: BookOrder | None = None
        if symbol is not None:
            order = book.get(symbol, {}).pop(order_id, None)
            if book.get(symbol) == {}:
                book.pop(symbol, None)
        else:
            for sym, orders in list(book.items()):
                if order_id in orders:
                    order = orders.pop(order_id)
                    if not orders:
                        book.pop(sym, None)
                    break
        if self.latency:
            await asyncio.sleep(self.latency)
        latency = time.monotonic() - start
        if order:
            tib = (datetime.utcnow() - order.placed_at).total_seconds()
            filled_qty = order.qty - order.pending_qty
            status = "partial" if filled_qty > 0 and order.pending_qty > 0 else "canceled"
            return {
                "status": status,
                "order_id": order.order_id,
                "time_in_book": tib,
                "qty": filled_qty,
                "filled_qty": filled_qty,
                "pending_qty": order.pending_qty,
                "latency": latency,
            }
        return {"status": "canceled", "order_id": order_id, "latency": latency}

    # Helpers de estado para inspección
    def equity(self, mark_prices: dict[str, float] | None = None) -> float:
        eq = self.state.cash + self.state.realized_pnl
        marks = mark_prices or self.state.last_px
        for sym, p in self.state.pos.items():
            mark = marks.get(sym)
            if mark is not None:
                eq += p.qty * mark
        return eq
