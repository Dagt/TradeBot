"""Helpers to track paper trading open orders and metrics state."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Tuple


class PaperOrderManager:
    """Track open orders and emit consistent metrics events.

    The manager keeps per-order information required to mirror the behaviour of
    an exchange when running the paper trading runner.  It stores the remaining
    quantity for each live order together with the notional that should be
    considered locked.  Helper methods update the registry on order creation,
    fills and cancellations while emitting the metrics payloads expected by the
    UI.
    """

    def __init__(
        self,
        emit: Callable[[dict], None],
        exposure_fn: Callable[[str | None], float],
    ) -> None:
        self._emit = emit
        self._exposure = exposure_fn
        self.open_orders: Dict[str, Dict[str, dict]] = defaultdict(dict)
        self.locked_total: float = 0.0
        self._orders_logged: set[Tuple[str, str]] = set()

    # ------------------------------------------------------------------
    def total_remaining(self, symbol: str, side: str | None = None) -> float:
        """Return remaining quantity for ``symbol`` and optionally ``side``."""

        orders = self.open_orders.get(symbol, {})
        if not orders:
            return 0.0
        if side is None:
            return sum(float(o["remaining_qty"]) for o in orders.values())
        side = side.lower()
        return sum(
            float(o["remaining_qty"])
            for o in orders.values()
            if o["side"] == side
        )

    # ------------------------------------------------------------------
    def on_order(
        self,
        *,
        symbol: str,
        order_id: str | None,
        side: str,
        price: float,
        orig_qty: float,
        remaining_qty: float,
        pnl: float | None = None,
    ) -> dict:
        """Register a freshly placed order and emit the ``order`` event."""

        side_norm = side.lower()
        remaining = max(float(remaining_qty), 0.0)
        entry = {
            "side": side_norm,
            "price": float(price),
            "orig_qty": float(orig_qty),
            "remaining_qty": remaining,
            "notional_locked_usd": float(price) * remaining,
        }
        if order_id is not None and remaining > 0:
            self.open_orders[symbol][order_id] = entry
        elif order_id is not None:
            # Ensure registry does not retain completed orders
            self.open_orders.get(symbol, {}).pop(order_id, None)
            if not self.open_orders.get(symbol):
                self.open_orders.pop(symbol, None)
        self._recalc_locked_total()
        payload = {
            "event": "order",
            "side": side,
            "price": float(price),
            "qty": float(orig_qty),
            "fee": 0.0,
        }
        if pnl is not None:
            payload["pnl"] = float(pnl)
        self._emit(payload)
        self._emit_exposure(symbol)
        if order_id is not None:
            self._orders_logged.add((symbol, order_id))
        return entry

    # ------------------------------------------------------------------
    def on_fill(
        self,
        *,
        symbol: str,
        order_id: str | None,
        side: str,
        fill_qty: float,
        price: float | None,
        fee: float | None,
        pending_qty: float | None,
        maker: bool | None = None,
        slippage_bps: float | None = None,
        pnl: float | None = None,
    ) -> dict:
        """Update the registry after a fill and emit a ``fill`` event."""

        side_norm = side.lower()
        entry = None
        if order_id is not None:
            entry = self.open_orders.get(symbol, {}).get(order_id)
        if entry is None:
            entry = {
                "side": side_norm,
                "price": float(price or 0.0),
                "orig_qty": float(fill_qty),
                "remaining_qty": 0.0,
                "notional_locked_usd": 0.0,
            }
            if order_id is not None:
                self.open_orders.setdefault(symbol, {})[order_id] = entry
        if price is not None:
            entry["price"] = float(price)
        remaining = entry.get("remaining_qty", 0.0)
        if pending_qty is None:
            remaining = max(remaining - float(fill_qty), 0.0)
        else:
            remaining = max(float(pending_qty), 0.0)
        entry["remaining_qty"] = remaining
        entry["notional_locked_usd"] = entry.get("price", 0.0) * remaining
        if order_id is not None and remaining <= 1e-9:
            self.open_orders.get(symbol, {}).pop(order_id, None)
            if not self.open_orders.get(symbol):
                self.open_orders.pop(symbol, None)
            self._orders_logged.discard((symbol, order_id))
        self._recalc_locked_total()
        payload = {
            "event": "fill",
            "side": side,
            "price": float(price) if price is not None else None,
            "qty": float(fill_qty),
            "fee": 0.0 if fee is None else float(fee),
        }
        if pnl is not None:
            payload["pnl"] = float(pnl)
        if slippage_bps is not None:
            payload["slippage_bps"] = float(slippage_bps)
        if maker is not None:
            payload["maker"] = bool(maker)
        self._emit(payload)
        self._emit_exposure(symbol)
        return entry

    # ------------------------------------------------------------------
    def on_cancel(
        self,
        *,
        symbol: str,
        order_id: str | None,
        reason: str | None = None,
    ) -> dict | None:
        """Remove an order from the registry and emit ``cancel`` metrics."""

        entry = None
        if order_id is not None:
            entry = self.open_orders.get(symbol, {}).pop(order_id, None)
            if not self.open_orders.get(symbol):
                self.open_orders.pop(symbol, None)
            self._orders_logged.discard((symbol, order_id))
        self._recalc_locked_total()
        payload = {"event": "cancel"}
        if reason:
            payload["reason"] = reason
        self._emit(payload)
        self._emit_exposure(symbol)
        return entry

    # ------------------------------------------------------------------
    def emit_skip(self, reason: str) -> None:
        """Emit a skip metric with ``reason``."""

        self._emit({"event": "skip", "reason": reason})

    # ------------------------------------------------------------------
    def ensure_order_event(
        self,
        *,
        symbol: str,
        order_id: str,
        side: str,
        price: float,
        orig_qty: float,
        remaining_qty: float,
    ) -> None:
        """Ensure the ``order`` event has been emitted for ``order_id``."""

        key = (symbol, order_id)
        if key in self._orders_logged:
            return
        self.on_order(
            symbol=symbol,
            order_id=order_id,
            side=side,
            price=price,
            orig_qty=orig_qty,
            remaining_qty=remaining_qty,
        )

    # ------------------------------------------------------------------
    def _emit_exposure(self, symbol: str | None) -> None:
        exposure = 0.0
        if symbol is not None:
            try:
                exposure = float(self._exposure(symbol))
            except Exception:
                exposure = 0.0
        self._emit({"exposure": exposure, "locked": self.locked_total})

    # ------------------------------------------------------------------
    def _recalc_locked_total(self) -> None:
        total = 0.0
        for orders in self.open_orders.values():
            for entry in orders.values():
                try:
                    total += float(entry.get("notional_locked_usd", 0.0))
                except (TypeError, ValueError):
                    continue
        if abs(total) <= 1e-9:
            total = 0.0
        self.locked_total = total


__all__ = ["PaperOrderManager"]
