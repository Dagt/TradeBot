from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import settings
from ..execution.order_types import Order
from ..utils.metrics import ORDERS, SKIPS, FILL_COUNT, CANCELS


_FINAL_STATUSES = {"filled", "expired", "cancelled", "canceled"}
_CANCEL_STATUSES = {"cancelled", "canceled", "expired", "partial"}


@dataclass
class _PaperOrderTracker:
    """Track asynchronous fills emitted by ``PaperAdapter``."""

    qty: float
    pending_qty: float | None = None
    status: str | None = None
    filled_qty: float = 0.0
    last_res: dict[str, Any] | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)

    def update(self, res: dict[str, Any]) -> None:
        """Update tracker state from an adapter response or fill event."""

        self.last_res = dict(res)
        status = res.get("status")
        if status is not None:
            self.status = str(status).lower()
        pending = res.get("pending_qty")
        if pending is not None:
            try:
                self.pending_qty = max(float(pending), 0.0)
            except (TypeError, ValueError):
                pass
        elif self.pending_qty is None:
            qty_val = res.get("qty")
            if qty_val is None:
                qty_val = res.get("filled_qty")
            try:
                qty_float = float(qty_val) if qty_val is not None else None
            except (TypeError, ValueError):
                qty_float = None
            if qty_float is not None and self.status == "filled":
                self.pending_qty = max(self.qty - qty_float, 0.0)
        if self.pending_qty is not None:
            self.filled_qty = max(self.qty - self.pending_qty, 0.0)
        if self.status in _FINAL_STATUSES or (
            self.pending_qty is not None and self.pending_qty <= 0
        ):
            self.event.set()


class Broker:
    """Simple broker wrapping an exchange adapter.

    Provides convenience helpers for limit orders with basic time-in-force
    handling, partial fills and re-quoting through optional callbacks.
    """

    def __init__(self, adapter, maker_fee_bps: float | None = None):
        self.adapter = adapter
        self.maker_fee_bps = (
            float(maker_fee_bps)
            if maker_fee_bps is not None
            else getattr(adapter, "maker_fee_bps", settings.maker_fee_bps)
        )
        self.passive_rebate_bps = getattr(settings, "passive_rebate_bps", 0.0)
        def _has_paper_venue(obj: Any, visited: set[int] | None = None) -> bool:
            if visited is None:
                visited = set()
            obj_id = id(obj)
            if obj_id in visited:
                return False
            visited.add(obj_id)
            name = str(getattr(obj, "name", "") or "").lower()
            if name == "paper":
                return True
            adapters_attr = getattr(obj, "adapters", None)
            if not adapters_attr:
                return False
            if isinstance(adapters_attr, dict):
                children = adapters_attr.values()
            elif isinstance(adapters_attr, (list, tuple, set)):
                children = adapters_attr
            else:
                try:
                    children = adapters_attr.values()  # type: ignore[attr-defined]
                except AttributeError:
                    try:
                        children = list(adapters_attr)
                    except TypeError:
                        return False
            for child in children:
                try:
                    if _has_paper_venue(child, visited):
                        return True
                except Exception:
                    continue
            return False

        self._track_paper_orders = _has_paper_venue(adapter)
        self._paper_orders: dict[str, _PaperOrderTracker] = {}

    def _register_paper_tracker(
        self, order_id: str | int | None, qty: float, res: dict[str, Any]
    ) -> _PaperOrderTracker | None:
        if not self._track_paper_orders:
            return None
        if order_id is None:
            return None
        key = str(order_id)
        tracker = self._paper_orders.get(key)
        if tracker is None:
            tracker = _PaperOrderTracker(qty=qty, pending_qty=qty)
            self._paper_orders[key] = tracker
        elif tracker.pending_qty is None:
            tracker.pending_qty = qty
        tracker.update(res)
        return tracker

    def _update_paper_tracker(self, res: dict[str, Any]) -> None:
        if not self._track_paper_orders:
            return
        order_id = res.get("order_id")
        if not order_id:
            return
        tracker = self._paper_orders.get(str(order_id))
        if tracker is not None:
            tracker.update(res)

    def update_last_price(self, symbol: str, px: float) -> list[dict[str, Any]]:
        """Update last price on the underlying adapter if supported.

        Any fills returned by the adapter are forwarded to the execution
        callbacks and risk service while updating metrics. A list with the
        processed fill events is returned so callers can forward them to
        :meth:`ExecutionRouter.handle_paper_event`.
        """

        collected: list[dict[str, Any]] = []

        def _handle_fills(fills, venue: str | None = None) -> None:
            if not fills:
                return
            cb_pf = getattr(self.adapter, "on_partial_fill", None)
            cb_exp = getattr(self.adapter, "on_order_expiry", None)
            risk = getattr(self.adapter, "risk_service", None)
            for raw in fills:
                event = dict(raw)
                status = str(event.get("status", "")).lower()
                side = event.get("side")
                qty = float(event.get("qty") or event.get("filled_qty") or 0.0)
                price = event.get("price")
                if side and qty > 0:
                    FILL_COUNT.labels(symbol=symbol, side=side).inc()
                    if risk is not None:
                        try:
                            risk.on_fill(symbol, side, qty, price=price, venue=venue)
                        except Exception:
                            pass
                cb = cb_exp if status in _FINAL_STATUSES else cb_pf
                if cb is not None and side:
                    try:
                        order_qty = event.get("qty")
                        if order_qty is None:
                            order_qty = qty + float(event.get("pending_qty", 0.0))
                        order = Order(
                            symbol=symbol,
                            side=side,
                            type_="limit",
                            qty=float(order_qty),
                            price=price,
                        )
                        order.pending_qty = float(event.get("pending_qty", 0.0))
                        cb(order, event)
                    except Exception:
                        pass
                self._update_paper_tracker(event)
                collected.append(event)

        upd = getattr(self.adapter, "update_last_price", None)
        if callable(upd):
            fills = upd(symbol, px)
            _handle_fills(fills, getattr(self.adapter, "name", None))
        elif hasattr(self.adapter, "adapters"):
            for ad in getattr(self.adapter, "adapters", {}).values():
                upd = getattr(ad, "update_last_price", None)
                if callable(upd):
                    fills = upd(symbol, px)
                    _handle_fills(fills, getattr(ad, "name", None))
        return collected

    def equity(self) -> float:
        """Return account equity if the adapter exposes it."""
        eq = getattr(self.adapter, "equity", None)
        if callable(eq):
            try:
                return float(eq())
            except Exception:
                return 0.0
        bal = getattr(self.adapter, "account", None)
        if bal is not None and hasattr(bal, "equity"):
            return float(getattr(bal, "equity", 0.0))
        return 0.0

    async def place_limit(
        self,
        symbol: str,
        side: str,
        price: float,
        qty: float,
        tif: str = "GTC|PO",
        on_partial_fill: Optional[Callable[[Order, dict], str | None]] = None,
        on_order_expiry: Optional[Callable[[Order, dict], str | None]] = None,
        signal_ts: float | None = None,
        *,
        slip_bps: float | None = None,
    ) -> dict:
        """Place a limit order respecting ``tif`` semantics.

        Parameters
        ----------
        symbol: str
            Trading pair, e.g. ``"BTC/USDT"``.
        side: str
            ``"buy"`` or ``"sell"``.
        price: float
            Limit price.
        qty: float
            Order quantity.
        tif: str, optional
            Time-in-force flags. Supports ``GTC``/``IOC``/``FOK`` and optional
            ``PO`` for post-only.  A ``GTD:<seconds>`` flag will cancel and
            replace the remaining quantity after the given number of seconds.
        on_partial_fill: callable, optional
            Callback executed when an order is partially filled. Should return
            ``"re_quote"`` to place the remaining quantity again. Any other
            value skips re-quoting.
        on_order_expiry: callable, optional
            Callback executed when an order expires (``GTD``). Should return
            ``"re_quote"`` to place the remaining quantity again. Any other
            value cancels the remainder.

        Returns
        -------
        dict
            Dictionary with keys ``status``, ``filled_qty``, ``pending_qty``,
            ``order_id`` and ``price`` describing the final state of the order
            or fallback order.
        """

        def _same_callback(cb1, cb2) -> bool:
            if cb1 is cb2:
                return True
            if cb1 is None or cb2 is None:
                return False
            if inspect.ismethod(cb1) and inspect.ismethod(cb2):
                return cb1.__func__ is cb2.__func__ and cb1.__self__ is cb2.__self__
            return False

        time_in_force = "GTC"
        post_only = False
        expiry = None
        for part in str(tif).split("|"):
            p = part.strip().upper()
            if p in {"GTC", "IOC", "FOK"}:
                time_in_force = p
            elif p == "PO":
                post_only = True
            elif p.startswith("GTD"):
                time_in_force = "GTD"
                try:
                    expiry = float(p.split(":", 1)[1])
                except (IndexError, ValueError):
                    expiry = None

        filled = 0.0
        remaining = qty
        attempts = 0
        max_attempts = int(getattr(settings, "requote_attempts", 5))
        last_res: dict = {}

        order = Order(
            symbol=symbol,
            side=side,
            type_="limit",
            qty=qty,
            price=price,
            post_only=post_only,
            time_in_force=time_in_force,
        )

        while remaining > 0 and attempts < max_attempts:
            attempts += 1
            order_qty = remaining
            kwargs = dict(
                symbol=symbol,
                side=side,
                type_="limit",
                qty=order_qty,
                price=price,
                post_only=post_only,
                time_in_force=time_in_force,
            )
            if slip_bps is not None:
                kwargs["slip_bps"] = slip_bps
            if signal_ts is not None:
                sig = inspect.signature(self.adapter.place_order)
                params = sig.parameters
                if (
                    "signal_ts" in params
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
                ):
                    kwargs["signal_ts"] = signal_ts
            res = await self.adapter.place_order(**kwargs)
            if res.get("status") == "rejected":
                SKIPS.inc()
            else:
                ORDERS.inc()

            order_id = res.get("order_id")
            tracker = self._register_paper_tracker(order_id, order_qty, res)

            qty_filled = float(
                res.get("qty") or res.get("filled") or res.get("filled_qty") or 0.0
            )
            filled += qty_filled
            remaining = float(res.get("pending_qty", remaining - qty_filled))
            order.pending_qty = remaining
            res.setdefault("filled_qty", qty_filled)
            res.setdefault("pending_qty", remaining)
            last_res = res

            tracker_status = None
            if tracker is not None:
                if tracker.pending_qty is not None:
                    remaining = max(float(tracker.pending_qty), 0.0)
                filled = max(float(filled), float(qty) - remaining)
                order.pending_qty = remaining
                if tracker.last_res is not None:
                    last_res = dict(tracker.last_res)
                tracker_status = tracker.status
            else:
                filled = max(float(filled), float(qty) - remaining)

            status = tracker_status if tracker_status is not None else res.get("status")
            status_lc = str(status or "").lower()

            if tracker is not None and tracker.event.is_set():
                break

            if remaining <= 0 or status_lc in {"canceled", "rejected"}:
                break

            if qty_filled > 0 and remaining > 0:
                action = on_partial_fill(order, last_res) if on_partial_fill else None
                if action in {"re_quote", "requote", "re-quote"}:
                    if order_id is not None:
                        self._paper_orders.pop(str(order_id), None)
                    price = order.price or price
                    continue
                break

            if expiry is not None and status_lc not in {"canceled", "rejected"}:
                timed_out = True
                if tracker is not None:
                    try:
                        await asyncio.wait_for(tracker.event.wait(), timeout=expiry)
                        timed_out = False
                    except asyncio.TimeoutError:
                        timed_out = True
                else:
                    await asyncio.sleep(expiry)
                if tracker is not None:
                    if tracker.pending_qty is not None:
                        remaining = max(float(tracker.pending_qty), 0.0)
                    filled = max(float(filled), float(qty) - remaining)
                    order.pending_qty = remaining
                    if tracker.last_res is not None:
                        last_res = dict(tracker.last_res)
                    if tracker.status:
                        status_lc = tracker.status
                    if tracker.event.is_set() or tracker.status in _FINAL_STATUSES or remaining <= 0:
                        break
                if tracker is not None and not timed_out:
                    continue

                adapter_cb = None
                callback_same = False
                callback_executed = False
                captured_action = None
                if on_order_expiry is not None:
                    adapter_cb = getattr(self.adapter, "on_order_expiry", None)
                    if _same_callback(on_order_expiry, adapter_cb):
                        callback_same = True

                        def _proxy(_order, adapter_res):
                            nonlocal callback_executed, captured_action
                            callback_executed = True
                            captured_action = on_order_expiry(order, adapter_res)
                            return captured_action

                        setattr(self.adapter, "on_order_expiry", _proxy)
                cancellation_recorded = False
                try:
                    cancel_res = await self.adapter.cancel_order(order_id, symbol)
                    cancel_status = str(cancel_res.get("status") or "").lower()
                    if cancel_status in _CANCEL_STATUSES:
                        cancellation_recorded = True
                    if tracker is not None:
                        tracker.update(cancel_res)
                        if tracker.pending_qty is not None:
                            remaining = max(float(tracker.pending_qty), 0.0)
                        filled = max(float(filled), float(qty) - remaining)
                        last_res = dict(tracker.last_res or cancel_res)
                        if tracker.status:
                            status_lc = tracker.status
                            if tracker.status in _CANCEL_STATUSES or (
                                tracker.event.is_set()
                                and tracker.status in _FINAL_STATUSES
                                and tracker.status != "filled"
                            ):
                                cancellation_recorded = True
                    else:
                        filled = float(cancel_res.get("filled_qty", 0.0))
                        pending_default = (
                            order.pending_qty
                            if getattr(order, "pending_qty", None) is not None
                            else remaining
                        )
                        remaining = float(cancel_res.get("pending_qty", pending_default))
                        if remaining <= 0 and filled > 0:
                            cancel_res["status"] = "filled"
                        cancel_res.setdefault("pending_qty", remaining)
                        cancel_res.setdefault("filled_qty", filled)
                        cancel_res.setdefault("status", cancel_res.get("status", "canceled"))
                        last_res = cancel_res
                        status_lc = str(cancel_res.get("status", status_lc)).lower()
                        if status_lc in _CANCEL_STATUSES:
                            cancellation_recorded = True
                except Exception:
                    pass
                finally:
                    if callback_same:
                        setattr(self.adapter, "on_order_expiry", adapter_cb)

                if cancellation_recorded:
                    CANCELS.inc()

                if on_order_expiry is None:
                    action = "re_quote"
                elif callback_same and callback_executed:
                    action = captured_action
                else:
                    action = on_order_expiry(order, last_res) if on_order_expiry else None
                if action in {"re_quote", "requote", "re-quote"}:
                    price = order.price or price
                    if order_id is not None:
                        self._paper_orders.pop(str(order_id), None)
                    continue
                break

            # Nothing else to do (no expiry and not fully filled)
            break

        remaining = max(float(remaining), 0.0)
        filled = max(float(filled), float(qty) - remaining)
        result = dict(last_res)
        result.update(
            {
                "status": last_res.get("status"),
                "filled_qty": float(filled),
                "filled": float(filled),
                "pending_qty": remaining,
            }
        )
        result.setdefault("order_id", last_res.get("order_id"))
        price_val = result.get("price")
        if price_val is None:
            price_val = last_res.get("price", price)
        if price_val is None:
            price_val = price
        result["price"] = price_val
        final_oid = result.get("order_id")
        if final_oid is not None:
            tracker = self._paper_orders.get(str(final_oid))
            if tracker is not None and (tracker.event.is_set() or remaining <= 0):
                self._paper_orders.pop(str(final_oid), None)
        last_res = result
        return last_res

    async def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        signal_ts: float | None = None,
        *,
        slip_bps: float | None = None,
    ) -> dict:
        """Place a market order and track submission metrics."""

        kwargs = {
            "symbol": symbol,
            "side": side,
            "type_": "market",
            "qty": qty,
        }
        if slip_bps is not None:
            kwargs["slip_bps"] = slip_bps
        if signal_ts is not None:
            kwargs["signal_ts"] = signal_ts
        res = await self.adapter.place_order(**kwargs)
        if res.get("status") == "rejected":
            SKIPS.inc()
        else:
            ORDERS.inc()
        return res
