# src/tradingbot/execution/paper.py
from __future__ import annotations
import asyncio
import logging
import time
import math
from dataclasses import dataclass, field
from typing import (
    Dict,
    Optional,
    Iterable,
    AsyncIterator,
    AsyncIterable,
    Mapping,
    Any,
    Sequence,
)
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

PRICE_TOLERANCE = 1e-9

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
    depth_snapshot: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
    queue_pos: float = 0.0


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
        # Dynamic per-symbol slippage estimates used when book data is absent
        self.slip_bps_per_qty_by_symbol: Dict[str, float] = {}

    def _sanitize_book_levels(
        self, levels: Sequence[Sequence[Any]] | Iterable[Sequence[Any]] | None
    ) -> list[list[float]]:
        sanitized: list[list[float]] = []
        if levels is None:
            return sanitized
        for level in levels:
            try:
                price = float(level[0])
                qty = float(level[1])
            except (TypeError, ValueError, IndexError):
                continue
            if qty < 0:
                continue
            sanitized.append([price, qty])
        return sanitized

    def _resolve_book_snapshot(
        self, symbol: str, book: Mapping[str, Any] | None
    ) -> Mapping[str, Any] | None:
        """Return a sanitized depth snapshot combining ``book`` with cached data."""

        prev = self.state.depth_snapshot.get(symbol, {})
        if not isinstance(book, Mapping):
            return prev or None

        snapshot: dict[str, Any] = dict(prev)
        changed = False

        def _coerce_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _assign_price(dst_key: str, *keys: str) -> None:
            nonlocal changed
            for key in keys:
                if key not in book:
                    continue
                val = _coerce_float(book[key])
                if val is None:
                    continue
                snapshot[dst_key] = val
                snapshot[f"{dst_key}_px"] = val
                snapshot[f"{dst_key}_price"] = val
                changed = True
                return

        def _assign_size(dst_key: str, alias: str, *keys: str) -> None:
            nonlocal changed
            for key in keys:
                if key not in book:
                    continue
                val = _coerce_float(book[key])
                if val is None:
                    continue
                size = max(val, 0.0)
                snapshot[dst_key] = size
                snapshot[alias] = size
                changed = True
                return

        _assign_price("bid", "bid", "bid_px", "bid_price", "best_bid")
        _assign_price("ask", "ask", "ask_px", "ask_price", "best_ask")
        _assign_size("bid_size", "bid_qty", "bid_size", "bid_qty", "best_bid_size")
        _assign_size("ask_size", "ask_qty", "ask_size", "ask_qty", "best_ask_size")

        if hasattr(book, "get") and "volume" in book:
            vol = _coerce_float(book.get("volume"))
            if vol is not None and math.isfinite(vol):
                snapshot["volume"] = max(vol, 0.0)
                changed = True
            elif "volume" in snapshot:
                snapshot.pop("volume", None)
                changed = True
        elif "volume" in snapshot and "volume" in prev:
            snapshot.pop("volume", None)
            changed = True

        bids = self._sanitize_book_levels(book.get("bids"))
        asks = self._sanitize_book_levels(book.get("asks"))
        if bids:
            snapshot["bids"] = bids
            changed = True
        if asks:
            snapshot["asks"] = asks
            changed = True

        ts = book.get("ts") if hasattr(book, "get") else None
        if ts is not None:
            snapshot["ts"] = ts
            changed = True

        if changed or snapshot:
            self.state.depth_snapshot[symbol] = snapshot
            return snapshot
        return prev or None

    def _book_capacity(self, book: Mapping[str, Any] | None, key: str) -> float:
        if not isinstance(book, Mapping):
            return float("inf")
        raw = book.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return float("inf")
        if not math.isfinite(val):
            return float("inf")
        return max(val, 0.0)

    def _has_volume_proxy(self, book: Mapping[str, Any] | None) -> bool:
        if not isinstance(book, Mapping):
            return False
        for key in ("volume", "base_volume", "quote_volume", "turnover"):
            if key not in book:
                continue
            raw = book.get(key)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val) and val > 0:
                return True
        return False

    def _depth_capacity_from_snapshot(
        self,
        book: Mapping[str, Any] | None,
        side: str,
        levels_cache: dict[str, list[list[float]]] | None = None,
    ) -> float:
        if not isinstance(book, Mapping):
            return 0.0
        key = "ask_size" if side == "buy" else "bid_size"
        cap = self._book_capacity(book, key)
        if math.isfinite(cap) and cap > 0:
            return cap
        levels_key = "asks" if side == "buy" else "bids"
        levels: list[list[float]] | None = None
        if levels_cache is not None:
            levels = levels_cache.get(levels_key)
        if levels is None:
            levels = self._sanitize_book_levels(book.get(levels_key))
            if levels_cache is not None:
                levels_cache[levels_key] = levels
        if not levels:
            return 0.0
        total = 0.0
        for _, qty in levels:
            total += max(qty, 0.0)
        return total if total > 0 else 0.0

    def update_last_price(
        self,
        symbol: str,
        px: float,
        qty: float | None = None,
        book: Mapping[str, Any] | None = None,
    ):
        self.state.last_px[symbol] = px
        self.account.mark_price(symbol, px)
        snapshot = self._resolve_book_snapshot(symbol, book)
        return self._match_book(symbol, px, qty, snapshot)

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
        if book is None:
            book = self.state.depth_snapshot.get(symbol)
        orders = self.state.book.get(symbol)
        if not orders:
            return fills
        remaining = float("inf") if qty is None else float(qty)
        now = datetime.utcnow()
        # Available top-of-book liquidity if provided
        if book is not None:
            remaining_asks = self._book_capacity(book, "ask_size")
            remaining_bids = self._book_capacity(book, "bid_size")
        else:
            remaining_asks = remaining_bids = float("inf")
        levels_cache: dict[str, list[list[float]]] = {}
        has_volume_proxy = self._has_volume_proxy(book)
        depth_caps = {
            "buy": self._depth_capacity_from_snapshot(book, "buy", levels_cache),
            "sell": self._depth_capacity_from_snapshot(book, "sell", levels_cache),
        }
        if book is not None:
            if math.isinf(remaining_asks) and depth_caps["buy"] > 0:
                remaining_asks = depth_caps["buy"]
            if math.isinf(remaining_bids) and depth_caps["sell"] > 0:
                remaining_bids = depth_caps["sell"]
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
                book_side = remaining_bids
            else:
                avail = (
                    self.state.cash / (px * (1 + fee_bps / 10000.0))
                    if self.state.cash > 0
                    else 0.0
                )
                book_side = remaining_asks
            qty_cap = min(order.pending_qty, remaining, avail)
            if qty_cap <= 0:
                continue
            book_side_val = book_side
            depth_cap = depth_caps["buy" if order.side == "buy" else "sell"]
            if book is not None:
                if depth_cap > 0:
                    if not math.isfinite(book_side_val):
                        book_side_val = depth_cap
                    else:
                        book_side_val = min(book_side_val, depth_cap)
                if not math.isfinite(book_side_val):
                    book_side_val = float("inf")
                if book_side_val <= 0:
                    continue
            else:
                if not math.isfinite(book_side_val):
                    book_side_val = float("inf")
            if (
                self.slippage_model is not None
                and isinstance(book, Mapping)
                and has_volume_proxy
                and depth_cap > 0
            ):
                bar = dict(book)
                key = "bid_size" if order.side == "sell" else "ask_size"
                bar[key] = depth_cap
                px_slip, fill_qty, new_q = self.slippage_model.fill(
                    order.side, qty_cap, px, bar, queue_pos=order.queue_pos
                )
                order.queue_pos = new_q
                if fill_qty <= 0:
                    continue
                slip_bps = (
                    (px_slip - px) / px * 10000.0
                    if order.side == "buy"
                    else (px - px_slip) / px * 10000.0
                )
            else:
                fill_qty = min(qty_cap, book_side_val)
                if fill_qty <= 0:
                    continue
                px_slip, slip_bps = self._apply_slippage(
                    symbol, order.side, fill_qty, px, book
                )
            # Update dynamic slippage estimate based on available depth
            if book is not None:
                levels = None
                if order.side == "buy" and "asks" in book:
                    levels = book["asks"]  # type: ignore[index]
                elif order.side == "sell" and "bids" in book:
                    levels = book["bids"]  # type: ignore[index]
                if levels:
                    depth_bps = impact_by_depth(order.side, fill_qty, levels)
                    if depth_bps is not None and fill_qty > 0:
                        self.slip_bps_per_qty_by_symbol[symbol] = abs(depth_bps) / fill_qty
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
        symbol: str,
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
        has_liquidity_proxy = self._has_volume_proxy(book)

        if (
            self.slippage_model is not None
            and isinstance(book, Mapping)
            and has_liquidity_proxy
            and self._book_capacity(book, "ask_size" if side == "buy" else "bid_size") > 0
        ):
            adj = self.slippage_model.adjust(side, qty, px, book)
            if side == "buy":
                slip_bps = (adj - px) / px * 10000.0
            else:
                slip_bps = (px - adj) / px * 10000.0
            levels = None
            if side == "buy" and "asks" in book:
                levels = book["asks"]  # type: ignore[index]
            elif side == "sell" and "bids" in book:
                levels = book["bids"]  # type: ignore[index]
            if levels:
                depth_bps = impact_by_depth(side, qty, levels)
                if depth_bps is not None and qty > 0:
                    self.slip_bps_per_qty_by_symbol[symbol] = abs(depth_bps) / qty
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
                if qty > 0:
                    self.slip_bps_per_qty_by_symbol[symbol] = abs(slip_bps) / qty
                px_slipped = px * (1 + slip_bps / 10000.0)
                return px_slipped, slip_bps

        # Linear fallback ---------------------------------------------------
        rate = self.slip_bps_per_qty_by_symbol.get(symbol, self.slip_bps_per_qty)
        slip_bps = rate * qty
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
        book = self._resolve_book_snapshot(symbol, book)
        working_px = price if price is not None else last
        if post_only and type_.lower() == "limit":
            def _book_price(keys: tuple[str, ...]) -> float | None:
                if book is None:
                    return None
                for key in keys:
                    val = book.get(key) if hasattr(book, "get") else None
                    if val is None:
                        continue
                    try:
                        px = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(px):
                        return px
                return None

            try:
                limit_before = float(price) if price is not None else None
            except (TypeError, ValueError):
                limit_before = None
            if limit_before is None or not math.isfinite(limit_before):
                log.info(
                    "Rejecting post-only limit order for %s: missing limit", symbol
                )
                return {
                    "status": "rejected",
                    "reason": "post_only_no_price",
                    "order_id": order_id,
                    "trade_id": trade_id,
                }
            best_bid = _book_price(("bid", "bid_px", "bid_price", "best_bid"))
            best_ask = _book_price(("ask", "ask_px", "ask_price", "best_ask"))
            ref_price = best_ask if side == "buy" else best_bid
            if ref_price is None or not math.isfinite(ref_price):
                ref_price = float(last)
            if ref_price is None or not math.isfinite(ref_price):
                log.info(
                    "Rejecting post-only limit order for %s: no reference price (limit=%s)",
                    symbol,
                    f"{limit_before:.10g}" if math.isfinite(limit_before) else "nan",
                )
                return {
                    "status": "rejected",
                    "reason": "post_only_no_reference",
                    "order_id": order_id,
                    "trade_id": trade_id,
                }
            fmt = lambda v: f"{v:.10g}" if v is not None and math.isfinite(v) else "nan"
            new_limit = limit_before
            if side == "buy":
                threshold = ref_price - PRICE_TOLERANCE
                if not math.isfinite(threshold) or threshold <= 0.0:
                    log.info(
                        "Rejecting post-only buy order for %s: invalid threshold %s",
                        symbol,
                        fmt(threshold),
                    )
                    return {
                        "status": "rejected",
                        "reason": "post_only_invalid_reference",
                        "order_id": order_id,
                        "trade_id": trade_id,
                    }
                if limit_before > threshold:
                    new_limit = min(limit_before, threshold)
                    if new_limit <= 0.0 or new_limit >= ref_price:
                        log.info(
                            "Rejecting post-only buy order for %s: limit %s vs ask %s",
                            symbol,
                            fmt(limit_before),
                            fmt(ref_price),
                        )
                        return {
                            "status": "rejected",
                            "reason": "post_only_would_cross",
                            "order_id": order_id,
                            "trade_id": trade_id,
                        }
                    log.info(
                        "Adjusted post-only buy order for %s: limit %s -> %s (best ask %s)",
                        symbol,
                        fmt(limit_before),
                        fmt(new_limit),
                        fmt(ref_price),
                    )
            else:
                threshold = ref_price + PRICE_TOLERANCE
                if not math.isfinite(threshold):
                    log.info(
                        "Rejecting post-only sell order for %s: invalid threshold %s",
                        symbol,
                        fmt(threshold),
                    )
                    return {
                        "status": "rejected",
                        "reason": "post_only_invalid_reference",
                        "order_id": order_id,
                        "trade_id": trade_id,
                    }
                if limit_before < threshold:
                    new_limit = max(limit_before, threshold)
                    if new_limit <= 0.0 or new_limit <= ref_price:
                        log.info(
                            "Rejecting post-only sell order for %s: limit %s vs bid %s",
                            symbol,
                            fmt(limit_before),
                            fmt(ref_price),
                        )
                        return {
                            "status": "rejected",
                            "reason": "post_only_would_cross",
                            "order_id": order_id,
                            "trade_id": trade_id,
                        }
                    log.info(
                        "Adjusted post-only sell order for %s: limit %s -> %s (best bid %s)",
                        symbol,
                        fmt(limit_before),
                        fmt(new_limit),
                        fmt(ref_price),
                    )
            price = float(new_limit)
            working_px = price
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
            crosses_buy = side == "buy" and price is not None and price >= last
            crosses_sell = side == "sell" and price is not None and price <= last
            if not post_only and (crosses_buy or crosses_sell):
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
            qpos = 0.0
            if book is not None:
                q_key = "ask_size" if side == "buy" else "bid_size"
                qpos = float(book.get(q_key, 0.0))
            order = BookOrder(
                order_id,
                symbol,
                side,
                float(price),
                qty,
                qty,
                datetime.utcnow(),
                timeout,
                qpos,
            )
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
            book_avail = self._book_capacity(book, "bid_size") if book else float("inf")
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
            book_avail = self._book_capacity(book, "ask_size") if book else float("inf")
            qty = min(qty, book_avail)
            if qty <= 0:
                return {
                    "status": "rejected",
                    "reason": "insufficient_cash",
                    "order_id": order_id,
                    "trade_id": trade_id,
                }
        px_exec, slip_bps = self._apply_slippage(symbol, side, qty, px_exec, book)
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
