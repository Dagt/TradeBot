"""Execution router with venue selection and basic metrics."""

from __future__ import annotations

import asyncio
import time
import inspect
import uuid
import contextlib
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, TYPE_CHECKING

from .order_types import Order
from .slippage import impact_by_depth, queue_position
from .normalize import adjust_order, SymbolRules, floor_to_step
from ..utils.metrics import (
    SLIPPAGE,
    ORDER_LATENCY,
    MAKER_TAKER_RATIO,
    FILL_COUNT,
    QUEUE_POSITION,
    SIGNAL_CONFIRM_LATENCY,
    ROUTER_SELECTED_VENUE,
    ROUTER_STALE_BOOK,
    SKIPS,
    CANCELS,
)
from ..storage import timescale
from ..utils.logging import get_logger
from ..config import settings

if TYPE_CHECKING:
    from ..risk.service import RiskService

log = get_logger(__name__)


class ExecutionRouter:
    """Route orders to the best venue and track execution metrics.

    A ``prefer`` flag can force maker or taker routing regardless of the
    order's ``post_only`` attribute.  When left as ``None`` the router keeps
    the previous behaviour of inferring maker vs taker from the order type.
    ``prefer`` accepts ``"maker"`` or ``"taker"``.
    """

    def __init__(
        self,
        adapters: Iterable,
        storage_engine=None,
        prefer: str | None = None,
        risk_service: "RiskService | None" = None,
        on_partial_fill=None,
        on_order_expiry=None,
    ):
        if isinstance(adapters, dict):
            self.adapters: Dict[str, object] = adapters  # type: ignore[assignment]
        elif isinstance(adapters, (list, tuple, set)):
            self.adapters = {
                getattr(ad, "name", f"venue{i}"): ad for i, ad in enumerate(adapters)
            }
        else:  # single adapter
            self.adapters = {getattr(adapters, "name", "default"): adapters}

        # Ensure adapters expose fee attributes for routing logic and cache
        # whether they support ``iceberg_qty`` to avoid repeated signature
        # inspections during order execution
        self._supports_iceberg_qty = {}
        for ad in self.adapters.values():
            if not hasattr(ad, "maker_fee_bps"):
                setattr(ad, "maker_fee_bps", getattr(settings, f"{ad.name}_maker_fee_bps", 0.0))
            if not hasattr(ad, "taker_fee_bps"):
                default = getattr(settings, f"{ad.name}_taker_fee_bps", getattr(ad, "maker_fee_bps", 0.0))
                setattr(ad, "taker_fee_bps", default)

            sig = inspect.signature(ad.place_order)
            params = sig.parameters
            self._supports_iceberg_qty[ad] = (
                "iceberg_qty" in params
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            )

        self._maker_counts = defaultdict(int)
        self._taker_counts = defaultdict(int)
        self._engine = storage_engine
        self._prefer = prefer
        self._last_maker = False
        self.risk_service = risk_service
        self.on_partial_fill = on_partial_fill
        self.on_order_expiry = on_order_expiry
        # Track live orders so PaperAdapter events can trigger callbacks
        self._active_orders: dict[str, tuple[Order, str, str, float]] = {}

        self._fee_update_task = None
        interval = getattr(settings, "router_fee_update_interval_sec", 3600.0)
        try:
            loop = asyncio.get_running_loop()
            self._fee_update_task = loop.create_task(self._fee_update_loop(interval))
        except RuntimeError:  # pragma: no cover - no running loop
            self._fee_update_task = None

    async def _fee_update_loop(self, interval: float) -> None:
        while True:
            await self.update_fees()
            await asyncio.sleep(interval)

    async def update_fees(self) -> None:
        for ad in self.adapters.values():
            if hasattr(ad, "update_fees"):
                try:
                    await ad.update_fees()  # type: ignore[attr-defined]
                except Exception as e:  # pragma: no cover - best effort
                    log.warning("update_fees failed: %s", e)

    async def close(self) -> None:
        task = getattr(self, "_fee_update_task", None)
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # ------------------------------------------------------------------
    async def place_order(
        self,
        symbol: str,
        side: str,
        type_: str,
        qty: float,
        price: float | None = None,
        *,
        post_only: bool = False,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        signal_ts: float | None = None,
        **kwargs,
    ) -> dict:
        """Build an :class:`Order` and forward to :meth:`execute`."""

        order = Order(
            symbol=symbol,
            side=side,
            type_=type_,
            qty=qty,
            price=price,
            post_only=post_only,
            time_in_force=time_in_force,
            iceberg_qty=kwargs.get("iceberg_qty"),
            reduce_only=reduce_only,
            reason=kwargs.get("reason"),
            slip_bps=kwargs.get("slip_bps"),
        )
        return await self.execute(order, signal_ts=signal_ts, **kwargs)

    # ------------------------------------------------------------------
    def _is_maker(self, order: Order) -> bool:
        maker = order.post_only or (
            order.type_.lower() == "limit" and order.time_in_force not in {"IOC", "FOK"}
        )
        if self._prefer == "maker":
            return True
        if self._prefer == "taker":
            return False
        return maker

    # ------------------------------------------------------------------
    async def best_venue(self, order: Order):
        """Select venue based on spread, fees, latency and slippage estimates."""
        maker_pref = self._is_maker(order)

        def venue_score(adapter) -> tuple[float, bool] | None:
            state = getattr(adapter, "state", None)
            book = getattr(state, "order_book", {}).get(order.symbol) if state else None
            ts = None
            if book:
                ts = book.get("ts") or book.get("timestamp")
                if ts is not None:
                    ts_val = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
                    age_ms = (time.time() - ts_val) * 1000.0
                    if age_ms > settings.router_max_book_age_ms:
                        ROUTER_STALE_BOOK.labels(
                            venue=getattr(adapter, "name", "unknown")
                        ).inc()
                        return None
            last = getattr(state, "last_px", {}).get(order.symbol) if state else None
            if book and book.get("bids") and book.get("asks"):
                bid = book["bids"][0][0]
                ask = book["asks"][0][0]
                spread = ask - bid
                price = ask if order.side == "buy" else bid
            elif last is not None:
                price = last
                spread = float("inf")
            else:
                return None

            maker = maker_pref
            fee_attr = "maker_fee_bps" if maker else "taker_fee_bps"
            fee_bps = float(
                getattr(adapter, fee_attr, getattr(adapter, "fee_bps", 0.0)) or 0.0
            )
            fee = price * fee_bps / 10000.0
            latency = getattr(adapter, "latency", 0.0)
            direction = 1 if order.side == "buy" else -1
            levels = (
                book["bids"] if maker and order.side == "buy"
                else book["asks"] if maker and order.side == "sell"
                else book["asks"] if order.side == "buy"
                else book["bids"]
            ) if book else []
            slip_bps = impact_by_depth(order.side, order.qty, levels) if not maker else None
            slip_px = price * (slip_bps or 0.0) / 10000.0
            queue_pos = queue_position(order.qty, levels) if maker else 0.0

            score = (
                settings.router_price_weight * direction * price
                + settings.router_fee_weight * fee
                + settings.router_spread_weight * spread / 2
                + settings.router_latency_weight * latency
                + settings.router_slippage_weight * direction * slip_px
                + settings.router_queue_weight * queue_pos
            )
            return score, maker

        selected = None
        best_score = None
        selected_maker = maker_pref
        for adapter in self.adapters.values():
            res = venue_score(adapter)
            if res is None:
                continue
            score, maker = res
            if best_score is None or score < best_score:
                best_score = score
                selected = adapter
                selected_maker = maker
        if selected is None:
            selected = next(iter(self.adapters.values()))
        self._last_maker = selected_maker
        ROUTER_SELECTED_VENUE.labels(
            venue=getattr(selected, "name", "unknown"),
            path="maker" if selected_maker else "taker",
        ).inc()
        return selected

    # ------------------------------------------------------------------
    async def execute(
        self,
        order: Order,
        algo: str | None = None,
        *,
        fill_mode: str = "close",
        slip_bps: float = 0.0,
        signal_ts: float | None = None,
        **algo_kwargs,
    ) -> dict | list[dict]:
        """Execute an order using optional execution algorithms.

        Parameters
        ----------
        order:
            Order to be routed.
        algo:
            Optional execution algorithm name (``twap``, ``vwap`` or ``pov``).
        **algo_kwargs:
            Additional parameters forwarded to the chosen algorithm.

        Returns
        -------
        dict | list[dict]
            Result(s) from the adapter or algorithm.
        """

        def _max_positive(*values: float | None) -> float | None:
            positives = [
                float(v) for v in values if v is not None and float(v) > 0.0
            ]
            return max(positives) if positives else None

        if self.risk_service is not None and not self.risk_service.enabled:
            log.warning("Risk manager disabled; rejecting order %s", order)
            SKIPS.inc()
            return {"status": "rejected", "reason": "risk_disabled"}

        if algo:
            from . import algos

            a = algo.lower()
            delay = float(algo_kwargs.get("delay", 0.0))
            if a == "twap":
                slices = int(algo_kwargs.get("slices", 1))
                children = algos.twap(order, slices)
            elif a == "vwap":
                volumes = algo_kwargs.get("volumes", [])
                children = algos.vwap(order, volumes)
            elif a == "pov":
                participation_rate = float(algo_kwargs["participation_rate"])
                trades = algo_kwargs.get("trades", [])
                children = algos.pov(order, trades, participation_rate)
            else:
                raise ValueError(f"unknown algo: {algo}")

            results: List[dict] = []
            for i, child in enumerate(children):
                res = await self.execute(child, signal_ts=signal_ts)
                results.append(res)
                if delay and i < len(children) - 1:
                    await asyncio.sleep(delay)
            return results

        if not hasattr(order, "_reserved_qty"):
            order._reserved_qty = (
                order.pending_qty if order.pending_qty is not None else order.qty
            )

        if (
            self.risk_service is not None
            and not self.risk_service.allow_short
            and order.side.lower() == "sell"
        ):
            cur_qty, _ = self.risk_service.account.current_exposure(order.symbol)
            if cur_qty <= 0:
                log.warning("Short not allowed; rejecting order %s", order)
                SKIPS.inc()
                return {"status": "rejected", "reason": "short_not_allowed"}
            if order.qty > cur_qty:
                order.qty = cur_qty
                order.pending_qty = cur_qty

        adapter = await self.best_venue(order)
        venue = getattr(adapter, "name", "unknown")
        log.info(
            "Routing order %s via %s reason=%s",
            order,
            venue,
            getattr(order, "reason", None),
        )

        maker = getattr(self, "_last_maker", self._is_maker(order))

        state = getattr(adapter, "state", None)
        book = getattr(state, "order_book", {}).get(order.symbol) if state else None

        rules: SymbolRules | None = None
        meta = getattr(adapter, "meta", None)
        if meta is not None:
            try:
                rules = meta.rules_for(order.symbol)
            except Exception:
                rules = None
        mark_price = order.price
        if mark_price is None:
            if state and getattr(state, "last_px", {}).get(order.symbol) is not None:
                mark_price = float(state.last_px[order.symbol])
            elif book and book.get("bids") and book.get("asks"):
                bid = book["bids"][0][0]
                ask = book["asks"][0][0]
                mark_price = (bid + ask) / 2.0
            else:
                mark_price = 0.0
        if rules is not None:
            adj = adjust_order(order.price, order.qty, float(mark_price or 0.0), rules, order.side)
            if not adj.ok:
                log.warning("Order %s rejected by adjust_order: %s", order, adj.reason)
                SKIPS.inc()
                return {"status": "rejected", "reason": adj.reason}
            order.price = adj.price
            order.qty = adj.qty
            order.pending_qty = adj.qty

        step_size = float(getattr(rules, "qty_step", 0.0) or 0.0) if rules else 0.0
        min_qty_rule = (
            float(rules.min_qty)
            if rules is not None and rules.min_qty is not None and rules.min_qty > 0
            else None
        )
        min_notional_rule = (
            float(rules.min_notional)
            if rules is not None and rules.min_notional is not None and rules.min_notional > 0
            else None
        )
        risk_min_qty = None
        risk_min_notional = None
        if self.risk_service is not None:
            risk_min_qty = getattr(self.risk_service, "min_order_qty", None)
            risk_min_notional = getattr(self.risk_service, "min_notional", None)

        min_qty_threshold = _max_positive(min_qty_rule, risk_min_qty)
        min_notional_threshold = _max_positive(min_notional_rule, risk_min_notional)

        if step_size:
            order.qty = floor_to_step(order.qty, step_size)
            order.pending_qty = order.qty

        price_for_notional = order.price if order.price is not None else mark_price
        price_for_notional = float(price_for_notional or 0.0)
        notional = abs(order.qty) * price_for_notional

        if abs(order.qty) <= 0.0:
            log.warning("Order %s rejected: qty<=0 tras cuantización", order)
            SKIPS.inc()
            return {"status": "rejected", "reason": "invalid_qty"}

        if min_qty_threshold and abs(order.qty) < min_qty_threshold:
            log.info(
                "Skipping order %s: qty %.8f below min %.8f",
                order,
                order.qty,
                min_qty_threshold,
            )
            SKIPS.inc()
            return {"status": "rejected", "reason": "below_min_qty"}

        if min_notional_threshold and notional < min_notional_threshold:
            log.info(
                "Skipping order %s: notional %.8f below min %.8f",
                order,
                notional,
                min_notional_threshold,
            )
            SKIPS.inc()
            return {"status": "rejected", "reason": "below_min_notional"}

        order._step_size = step_size
        order._min_qty_threshold = min_qty_threshold
        order._min_notional_threshold = min_notional_threshold
        order._price_for_notional = price_for_notional

        est_slippage = None
        queue_pos = None
        if book and book.get("bids") and book.get("asks"):
            levels: List[Tuple[float, float]] = (
                book["bids"] if maker and order.side == "buy" else
                book["asks"] if maker and order.side == "sell" else
                book["asks"] if order.side == "buy" else book["bids"]
            )
            if maker:
                queue_pos = queue_position(order.qty, levels)
                QUEUE_POSITION.labels(symbol=order.symbol, side=order.side).observe(queue_pos)
            else:
                est_slippage = impact_by_depth(order.side, order.qty, levels)

        start = time.monotonic()
        kwargs = dict(
            symbol=order.symbol,
            side=order.side,
            type_=order.type_,
            qty=order.qty,
            price=order.price,
            post_only=order.post_only,
            time_in_force=order.time_in_force,
            reduce_only=order.reduce_only,
        )
        if order.iceberg_qty is not None and self._supports_iceberg_qty.get(adapter):
            kwargs["iceberg_qty"] = order.iceberg_qty
        timeout = getattr(order, "timeout", None)
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            res = await adapter.place_order(**kwargs)
        except Exception as exc:  # pragma: no cover - logging only
            log.exception("Order placement failed on %s", venue)
            SKIPS.inc()
            return {"status": "error", "reason": str(exc), "venue": venue}

        latency = time.monotonic() - start
        ORDER_LATENCY.labels(venue=venue).observe(latency)
        res.setdefault("est_slippage_bps", est_slippage)
        res.setdefault("queue_position", queue_pos)
        res.setdefault("order_id", uuid.uuid4().hex)
        res.setdefault("trade_id", uuid.uuid4().hex)
        res.setdefault("reason", getattr(order, "reason", None))
        fee_attr = "maker_fee_bps" if maker else "taker_fee_bps"
        fee_bps = float(
            getattr(adapter, fee_attr, getattr(adapter, "fee_bps", 0.0)) or 0.0
        )
        res.setdefault("fee_type", "maker" if maker else "taker")
        res.setdefault("fee_bps", fee_bps)
        status = res.get("status")
        if status == "rejected":
            SKIPS.inc()
        if status == "new":
            oid = res.get("order_id")
            if oid:
                self._active_orders[oid] = (order, venue, fill_mode, slip_bps)
        if status in {"partial", "expired", "cancelled"}:
            filled = float(res.get("filled_qty", 0.0))
            prev_pending = order.pending_qty if order.pending_qty is not None else order.qty
            remaining = max(prev_pending - filled, 0.0)
            step_pending = float(getattr(order, "_step_size", 0.0) or 0.0)
            if step_pending:
                remaining = floor_to_step(remaining, step_pending)
            if remaining < 0:
                remaining = 0.0
            order.pending_qty = remaining
            res["pending_qty"] = remaining

            price_for_remaining = order.price
            if price_for_remaining is None:
                price_for_remaining = (
                    res.get("price")
                    or res.get("fill_price")
                    or getattr(order, "_price_for_notional", None)
                    or mark_price
                )
            price_for_remaining = float(price_for_remaining or 0.0)
            order._price_for_notional = price_for_remaining
            remaining_notional = abs(remaining) * price_for_remaining

            min_qty_threshold = getattr(order, "_min_qty_threshold", None)
            min_notional_threshold = getattr(order, "_min_notional_threshold", None)
            cancel_reason = None
            if remaining > 0:
                if min_qty_threshold and remaining < min_qty_threshold:
                    cancel_reason = "below_min_qty"
                elif min_notional_threshold and remaining_notional < min_notional_threshold:
                    cancel_reason = "below_min_notional"

            if self.risk_service is not None:
                if status in {"expired", "cancelled"} or cancel_reason or remaining <= 0:
                    released = getattr(order, "_reserved_qty", 0.0)
                    if released:
                        self.risk_service.account.update_open_order(
                            order.symbol, order.side, -released
                        )
                    order._reserved_qty = 0.0
                else:
                    delta = remaining - getattr(order, "_reserved_qty", 0.0)
                    if delta:
                        self.risk_service.account.update_open_order(
                            order.symbol, order.side, delta
                        )
                    order._reserved_qty = remaining
                if filled:
                    book = self.risk_service.positions_multi.get(venue, {})
                    cur_qty = book.get(order.symbol, 0.0)
                    delta_pos = filled if order.side == "buy" else -filled
                    self.risk_service.update_position(
                        venue,
                        order.symbol,
                        cur_qty + delta_pos,
                        entry_price=res.get("price"),
                    )

            if cancel_reason:
                await self._cancel_residual_order(order, adapter, venue, cancel_reason, res)
                status = res.get("status", status)
                remaining = float(res.get("pending_qty", 0.0))

            if remaining <= 0 or status in {"expired", "cancelled"}:
                res["locked"] = 0.0

            cb = self.on_partial_fill if status == "partial" else self.on_order_expiry
            action = cb(order, res) if cb else None
            if action in {"re_quote", "re-quote", "requote"} and order.pending_qty > 0:
                new_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    type_=order.type_,
                    qty=order.pending_qty,
                    price=order.price,
                    post_only=order.post_only,
                    time_in_force=order.time_in_force,
                    reduce_only=order.reduce_only,
                    reason=getattr(order, "reason", None),
                )
                new_order._reserved_qty = getattr(order, "_reserved_qty", 0.0)
                res2 = await self.execute(
                    new_order, fill_mode=fill_mode, slip_bps=slip_bps
                )
                order.pending_qty = res2.get("pending_qty", order.pending_qty)
                return res2
            res.setdefault("venue", venue)
            return res
        if res.get("status") == "filled":
            fill_price = res.get("price")
            if fill_mode == "bidask" and book and book.get("bids") and book.get("asks"):
                bid = book["bids"][0][0]
                ask = book["asks"][0][0]
                mid = (bid + ask) / 2.0
                spread = ask - bid
                fill_price = mid + spread / 2.0 if order.side == "buy" else mid - spread / 2.0
            if fill_price is not None and slip_bps:
                slip_mult = slip_bps / 10000.0
                if order.side == "buy":
                    fill_price *= 1 + slip_mult
                else:
                    fill_price *= 1 - slip_mult
            if fill_price is not None:
                res["price"] = fill_price
                res["fill_price"] = fill_price

            FILL_COUNT.labels(symbol=order.symbol, side=order.side).inc()
            if self._engine is not None:
                try:
                    timescale.insert_order(
                        self._engine,
                        strategy="router",
                        exchange=venue,
                        symbol=order.symbol,
                        side=order.side,
                        type_=order.type_,
                        qty=float(order.qty),
                        px=res.get("price"),
                        status=res.get("status", "unknown"),
                        ext_order_id=res.get("order_id"),
                        notes={
                            "fee_type": "maker" if maker else "taker",
                            "fee_bps": fee_bps,
                            "reduce_only": order.reduce_only,
                        },
                    )
                except Exception:  # pragma: no cover - logging only
                    log.exception("Persist failure: insert_order")
            log.info(
                "Order filled venue=%s oid=%s tid=%s reason=%s",
                venue,
                res.get("order_id"),
                res.get("trade_id"),
                res.get("reason"),
            )

        base_price = res.get("fill_price") or res.get("price")
        slip_bps = getattr(order, "slip_bps", None)
        if base_price is not None:
            fp = float(base_price)
            if slip_bps:
                adj = fp * float(slip_bps) / 10000.0
                fp = fp + adj if order.side == "buy" else fp - adj
            res["fill_price"] = fp
        exec_price = res.get("price")
        expected = order.price
        if expected is None:
            expected = getattr(getattr(adapter, "state", None), "last_px", {}).get(order.symbol)
        if exec_price is not None and expected is not None:
            bps = (exec_price - expected) / expected * 10000
            if bps != 0:
                SLIPPAGE.labels(symbol=order.symbol, side=order.side).observe(bps)

        if maker:
            self._maker_counts[venue] += 1
        else:
            self._taker_counts[venue] += 1
        maker_c = self._maker_counts[venue]
        taker_c = self._taker_counts[venue]
        ratio = maker_c / taker_c if taker_c else float(maker_c)
        MAKER_TAKER_RATIO.labels(venue=venue).set(ratio)
        # Propagate venue so downstream components can track positions per exchange
        res.setdefault("venue", venue)
        filled_flag = float(res.get("pending_qty", 0.0)) <= 0.0
        log.info(
            "Order placed venue=%s oid=%s tid=%s reason=%s filled=%s",
            venue,
            res.get("order_id"),
            res.get("trade_id"),
            res.get("reason"),
            filled_flag,
        )
        if signal_ts is not None:
            SIGNAL_CONFIRM_LATENCY.observe(time.time() - signal_ts)
        if self.risk_service is not None:
            filled = float(res.get("filled_qty") or res.get("qty") or 0.0)
            pending = float(res.get("pending_qty", 0.0))
            delta = pending - getattr(order, "_reserved_qty", 0.0)
            if delta:
                self.risk_service.account.update_open_order(
                    order.symbol, order.side, delta
                )
            order._reserved_qty = pending
            if filled:
                book = self.risk_service.positions_multi.get(venue, {})
                cur_qty = book.get(order.symbol, 0.0)
                delta_pos = filled if order.side == "buy" else -filled
                self.risk_service.update_position(
                    venue,
                    order.symbol,
                    cur_qty + delta_pos,
                    entry_price=res.get("price"),
                )
        return res

    # ------------------------------------------------------------------
    async def _cancel_residual_order(
        self,
        order: Order,
        adapter,
        venue: str,
        reason: str,
        res: dict,
    ) -> None:
        oid = res.get("order_id")
        cancel_details: dict | None = None
        if adapter is not None and hasattr(adapter, "cancel_order") and oid:
            try:
                cancel_details = await adapter.cancel_order(oid, order.symbol)
            except Exception as exc:  # pragma: no cover - best effort
                log.warning(
                    "Residual cancel failed venue=%s oid=%s reason=%s error=%s",
                    venue,
                    oid,
                    reason,
                    exc,
                )
            else:
                CANCELS.inc()
        if oid:
            self._active_orders.pop(oid, None)
        order.pending_qty = 0.0
        order._reserved_qty = 0.0
        res["status"] = "cancelled"
        res["pending_qty"] = 0.0
        res["cancel_reason"] = reason
        res["locked"] = 0.0
        if cancel_details:
            res.setdefault("cancel_details", cancel_details)
            if (
                cancel_details.get("filled_qty") is not None
                and res.get("filled_qty") is None
            ):
                res["filled_qty"] = cancel_details["filled_qty"]
        log.info(
            "Residual order cancelled venue=%s oid=%s reason=%s",
            venue,
            oid,
            reason,
        )

    # ------------------------------------------------------------------
    async def cancel_order(
        self, order_id: str, *, venue: str | None = None, symbol: str | None = None
    ) -> dict:
        """Cancel an existing order and update metrics once.

        The tracked order is removed immediately to avoid processing
        subsequent adapter events and double-counting metrics.
        """

        info = self._active_orders.pop(order_id, None)
        if venue is None and info is not None:
            _, venue, _, _ = info
        adapter = None
        if venue is not None:
            adapter = self.adapters.get(venue)
        if adapter is None:
            adapter = next(iter(self.adapters.values()))
        try:
            res = await adapter.cancel_order(order_id, symbol)
        except Exception as e:  # pragma: no cover - best effort
            log.error("Cancel order failed: %s", e)
            res = {"status": "error"}
        CANCELS.inc()
        if info and self.risk_service is not None:
            order, _, _, _ = info
            released = getattr(order, "_reserved_qty", 0.0)
            if released:
                self.risk_service.account.update_open_order(
                    order.symbol, order.side, -released
                )
            order._reserved_qty = 0.0
            if self.on_order_expiry is not None:
                try:
                    self.on_order_expiry(order, res)
                except Exception:  # pragma: no cover - best effort
                    pass
        log.info("Order cancelled venue=%s oid=%s", getattr(adapter, "name", "unknown"), order_id)
        return res

    # ------------------------------------------------------------------
    async def handle_paper_event(self, res: dict) -> dict | None:
        """Process fill/expiry events from ``PaperAdapter``.

        The router keeps track of live orders placed through ``execute``. When
        the underlying :class:`PaperAdapter` later emits events via
        ``update_last_price`` this method updates order state, forwards the
        appropriate callbacks and optionally re-quotes remaining quantity.
        """

        oid = res.get("order_id")
        if not oid:
            return None
        info = self._active_orders.get(oid)
        if info is None:
            return None
        order, venue, fill_mode, slip_bps = info
        status = res.get("status")
        if status not in {"partial", "expired", "filled"}:
            return None

        adapter = self.adapters.get(venue)
        filled = float(res.get("filled_qty", 0.0))
        prev_pending = order.pending_qty if order.pending_qty is not None else order.qty
        remaining = max(prev_pending - filled, 0.0)
        step_pending = float(getattr(order, "_step_size", 0.0) or 0.0)
        if step_pending:
            remaining = floor_to_step(remaining, step_pending)
        if remaining < 0:
            remaining = 0.0
        order.pending_qty = remaining
        res["pending_qty"] = remaining

        price_for_remaining = order.price
        if price_for_remaining is None:
            price_for_remaining = (
                res.get("price")
                or res.get("fill_price")
                or getattr(order, "_price_for_notional", None)
            )
        price_for_remaining = float(price_for_remaining or 0.0)
        order._price_for_notional = price_for_remaining
        remaining_notional = abs(remaining) * price_for_remaining

        min_qty_threshold = getattr(order, "_min_qty_threshold", None)
        min_notional_threshold = getattr(order, "_min_notional_threshold", None)
        cancel_reason = None
        if status == "partial" and remaining > 0:
            if min_qty_threshold and remaining < min_qty_threshold:
                cancel_reason = "below_min_qty"
            elif min_notional_threshold and remaining_notional < min_notional_threshold:
                cancel_reason = "below_min_notional"

        if self.risk_service is not None:
            if status in {"expired", "filled"} or cancel_reason or remaining <= 0:
                released = getattr(order, "_reserved_qty", 0.0)
                if released:
                    self.risk_service.account.update_open_order(
                        order.symbol, order.side, -released
                    )
                order._reserved_qty = 0.0
            else:
                delta = remaining - getattr(order, "_reserved_qty", 0.0)
                if delta:
                    self.risk_service.account.update_open_order(
                        order.symbol, order.side, delta
                    )
                order._reserved_qty = remaining
            if filled:
                book = self.risk_service.positions_multi.get(venue, {})
                cur_qty = book.get(order.symbol, 0.0)
                delta_pos = filled if order.side == "buy" else -filled
                self.risk_service.update_position(
                    venue,
                    order.symbol,
                    cur_qty + delta_pos,
                    entry_price=res.get("price"),
                )

        if cancel_reason:
            await self._cancel_residual_order(order, adapter, venue, cancel_reason, res)
            status = res.get("status", status)
            remaining = float(res.get("pending_qty", 0.0))

        if remaining <= 0 or status in {"expired", "filled", "cancelled"}:
            res["locked"] = 0.0

        cb = self.on_partial_fill if status == "partial" else self.on_order_expiry
        action = cb(order, res) if cb else None
        if action in {"re_quote", "re-quote", "requote"} and order.pending_qty > 0:
            self._active_orders.pop(oid, None)
            new_order = Order(
                symbol=order.symbol,
                side=order.side,
                type_=order.type_,
                qty=order.pending_qty,
                price=order.price,
                post_only=order.post_only,
                time_in_force=order.time_in_force,
                reduce_only=order.reduce_only,
                reason=getattr(order, "reason", None),
            )
            new_order._reserved_qty = getattr(order, "_reserved_qty", 0.0)
            res2 = await self.execute(
                new_order, fill_mode=fill_mode, slip_bps=slip_bps
            )
            return res2

        if status in {"expired", "filled", "cancelled"} or order.pending_qty <= 0:
            self._active_orders.pop(oid, None)
        res.setdefault("venue", venue)
        return res


__all__ = ["ExecutionRouter"]

