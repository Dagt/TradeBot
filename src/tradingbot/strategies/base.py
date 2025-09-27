from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import math
import re
import time

import yaml

from ..utils.metrics import REQUEST_LATENCY
from ..storage import timescale
from ..execution.order_types import Order
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class Signal:
    """Simple trading signal.

    ``strength`` defines sizing through ``notional = equity * strength``. A
    value of ``1.0`` uses the full account equity, values above ``1.0`` pyramid
    exposure and values between ``0`` and ``1`` scale it down. ``0`` or
    negative values close the position.
    """

    side: str  # 'buy' | 'sell' | 'flat'
    strength: float = 1.0  # fraction of the base equity allocation
    reduce_only: bool = False
    limit_price: float | None = None
    post_only: bool | None = None
    signal_ts: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

def timeframe_to_minutes(tf: str | int | float | None) -> float:
    """Return the timeframe duration expressed in minutes.

    Parameters
    ----------
    tf:
        Timeframe expressed as strings like ``"1m"`` or ``"1h"``.  Numerical
        values are interpreted directly as minutes.  When ``None`` is passed
        the function falls back to ``1`` minute.
    """

    if tf is None:
        return 1.0
    if isinstance(tf, (int, float)):
        val = float(tf)
        return 1.0 if val <= 0 else val
    tf_str = str(tf).strip().lower()
    if not tf_str:
        return 1.0
    if tf_str.isdigit():
        val = float(tf_str)
        return 1.0 if val <= 0 else val
    m = re.fullmatch(r"(\d+)([smhd])", tf_str)
    if not m:
        return 1.0
    value = float(m.group(1))
    unit = m.group(2)
    factors = {"s": 1.0 / 60.0, "m": 1.0, "h": 60.0, "d": 1440.0}
    factor = factors.get(unit, 1.0)
    minutes = value * factor
    return 1.0 if minutes <= 0 else minutes


class Strategy(ABC):
    name: str
    max_signal_strength: float = 3.0
    min_signal_strength: float = 0.3
    min_strength_fraction: float = 0.1

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...

    # ------------------------------------------------------------------
    def _requote_key(self, order: Order) -> tuple[str | None, str | None]:
        return getattr(order, "symbol", None), getattr(order, "side", None)
      
    def _track_requote(
        self,
        order: Order | None = None,
        *,
        key: tuple[str | None, str | None] | None = None,
        reset: bool = False,
    ) -> int:
        if not hasattr(self, "_requote_attempts"):
            self._requote_attempts: dict[tuple[str | None, str | None], int] = {}
        if key is None:
            if order is None:
                return 0
            key = self._requote_key(order)
        if reset:
            self._requote_attempts[key] = 0
        else:
            self._requote_attempts[key] = self._requote_attempts.get(key, 0) + 1
        return self._requote_attempts.get(key, 0)

    def _reprice_order(self, order: Order, res: dict[str, Any]) -> None:
        """Update ``order.price`` using metadata from the last signal."""

        last_sig = getattr(self, "_last_signal", {}).get(order.symbol)
        if last_sig is None:
            return
        meta = getattr(last_sig, "metadata", None) or {}
        base_price = meta.get("base_price")
        if base_price is None:
            base_price = order.price if order.price is not None else res.get("price")
        try:
            base_price = float(base_price)
        except (TypeError, ValueError):
            return
        offset = meta.get("limit_offset")
        if offset is None:
            offset = 0.0
        try:
            offset = abs(float(offset))
        except (TypeError, ValueError):
            offset = 0.0
        initial_offset = meta.get("initial_offset")
        try:
            initial_offset = abs(float(initial_offset)) if initial_offset is not None else None
        except (TypeError, ValueError):
            initial_offset = None
        patience = meta.get("maker_patience")
        try:
            patience = int(patience) if patience is not None else None
        except (TypeError, ValueError):
            patience = None
        if patience is not None and patience < 0:
            patience = 0
        step_offset = meta.get("offset_step")
        try:
            step_offset = abs(float(step_offset)) if step_offset is not None else None
        except (TypeError, ValueError):
            step_offset = None
        maker_initial = meta.get("maker_initial_offset")
        try:
            maker_initial = abs(float(maker_initial)) if maker_initial is not None else None
        except (TypeError, ValueError):
            maker_initial = None
        if offset == 0.0 and step_offset is None and initial_offset is None:
            atr_map = getattr(self, "_last_atr", {})
            atr_val = None
            if isinstance(atr_map, dict):
                atr_val = atr_map.get(order.symbol)
            if atr_val is not None:
                try:
                    offset = 0.1 * abs(float(atr_val))
                except (TypeError, ValueError):
                    offset = 0.0
            if offset == 0.0 and order.price is not None:
                try:
                    offset = abs(float(order.price)) * 0.0005
                except (TypeError, ValueError):
                    offset = 0.0
        max_offset = meta.get("max_offset")
        try:
            max_offset = abs(float(max_offset)) if max_offset is not None else None
        except (TypeError, ValueError):
            max_offset = None
        step_mult = meta.get("step_mult", 0.5)
        try:
            step_mult = float(step_mult)
        except (TypeError, ValueError):
            step_mult = 0.5
        attempts = 0
        if hasattr(self, "_requote_attempts"):
            attempts = self._requote_attempts.get(self._requote_key(order), 0)
        direction = 1.0 if str(order.side).lower() == "buy" else -1.0
        chase = bool(meta.get("chase", True))
        default_requote = "limit_offset" not in meta or not meta
        if chase:
            if initial_offset is not None or step_offset is not None:
                init = initial_offset if initial_offset is not None else offset
                if init is None:
                    init = 0.0
                step_inc = step_offset if step_offset is not None else offset
                if step_inc is None:
                    step_inc = 0.0
                attempt_idx = max(1, attempts)
                if patience is not None:
                    patience_idx = max(0, patience)
                    if attempt_idx <= patience_idx:
                        step = init
                    else:
                        progress = attempt_idx - patience_idx
                        first_step = maker_initial
                        if first_step is None:
                            first_step = init
                        if first_step is None:
                            first_step = 0.0
                        step = first_step
                        if progress > 1:
                            step += (progress - 1) * step_inc
                else:
                    if attempt_idx <= 1:
                        step = init
                    else:
                        step = init + (attempt_idx - 1) * step_inc
                if offset:
                    step = min(step, offset)
                if max_offset is not None:
                    step = min(step, max_offset)
            else:
                multiplier = 1.0 if default_requote else max(1.0, 1.0 + step_mult * attempts)
                step = offset * multiplier
                if max_offset is not None:
                    step = min(step, max_offset)
        else:
            decay = meta.get("decay", 0.5)
            try:
                decay = float(decay)
            except (TypeError, ValueError):
                decay = 0.5
            min_offset = meta.get("min_offset", 0.0)
            try:
                min_offset = abs(float(min_offset))
            except (TypeError, ValueError):
                min_offset = 0.0
            step = offset * (decay ** attempts)
            step = max(step, min_offset)
            if max_offset is not None:
                step = min(step, max_offset)
        tick_size = meta.get("tick_size")
        try:
            tick_size = float(tick_size) if tick_size is not None else None
        except (TypeError, ValueError):
            tick_size = None
        new_price = base_price + direction * step
        if tick_size and tick_size > 0:
            new_price = round(new_price / tick_size) * tick_size
        order.price = new_price

    # ------------------------------------------------------------------
    def _edge_still_exists(self, order: Order) -> bool:
        last = getattr(self, "_last_signal", {}).get(order.symbol)
        return bool(last and last.side == order.side and last.strength > 0.0)

    # ------------------------------------------------------------------
    def on_partial_fill(self, order: Order, res: dict[str, Any]):
        """Handle partial fills.

        Stores remaining ``pending_qty`` and decides whether to re-quote
        based on the current signal. When the trading edge disappears the
        default behaviour is to cancel the rest of the order.
        """

        if not hasattr(self, "pending_qty"):
            self.pending_qty: dict[str, float] = {}
        pending = float(res.get("pending_qty", 0.0))
        self.pending_qty[order.symbol] = pending
        if pending <= 0:
            return None
        if not self._edge_still_exists(order):
            return None
        attempts = self._track_requote(order)
        if attempts > 0:
            self._reprice_order(order, res)
        return "re_quote"

    # ------------------------------------------------------------------
    def on_order_expiry(self, order: Order, res: dict[str, Any]):
        """Handle order expiry events.

        Behaviour mirrors :meth:`on_partial_fill` where orders are
        re‑quoted only if the last signal still points in the same
        direction.  Expired orders are re‑submitted at the previous
        limit price plus a small ATR-based offset (≈0.1 × ATR) to
        improve the fill probability.  The offset is calculated
        internally and does not require additional parameters.
        """

        if not hasattr(self, "pending_qty"):
            self.pending_qty: dict[str, float] = {}
        pending = float(res.get("pending_qty", order.qty))
        self.pending_qty[order.symbol] = pending
        if pending <= 0:
            return None
        if not self._edge_still_exists(order):
            return None
        self._track_requote(order)
        self._reprice_order(order, res)
        return "re_quote"

    # ------------------------------------------------------------------
    def finalize_signal(
        self,
        bar: dict[str, Any],
        price: float,
        signal: Signal | None,
    ) -> Signal | None:
        """Attach limit prices and coordinate risk management.

        Strategies should call this helper at the end of ``on_bar`` passing the
        generated ``signal`` (which may be ``None``).  The function consults the
        shared ``RiskService`` to update trailing stops, evaluate opposite or
        stronger signals and decide whether the current trade should be closed
        or scaled.  When no ``RiskService`` is configured it simply attaches the
        ``limit_price`` and returns the signal unchanged.
        """

        symbol = bar.get("symbol")
        if signal is not None:
            if symbol:
                self._track_requote(key=(symbol, signal.side), reset=True)
            if signal.limit_price is None:
                signal.limit_price = price
        rs = getattr(self, "risk_service", None)
        if rs is None:
            return signal

        def _positive(value: Any) -> float | None:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(val) or val <= 0.0:
                return None
            return val

        volatility = None
        for key in ("atr", "volatility"):
            volatility = _positive(bar.get(key))
            if volatility is not None:
                break
        target_volatility = _positive(bar.get("target_volatility"))

        trade = rs.get_trade(symbol) if symbol else None
        if trade:
            atr = bar.get("atr") or bar.get("volatility")
            is_dict = isinstance(trade, dict)
            _missing = object()
            atr_prev = _missing
            if atr is not None:
                if is_dict:
                    atr_prev = trade.get("atr") if "atr" in trade else _missing
                    trade["atr"] = atr
                else:
                    atr_prev = getattr(trade, "atr") if hasattr(trade, "atr") else _missing
                    setattr(trade, "atr", atr)
            rs.update_trailing(trade, price)
            if is_dict:
                had_trail = "_trail_done" in trade
                trail_prev = trade.get("_trail_done") if had_trail else _missing
                trade["_trail_done"] = True
            else:
                had_trail = hasattr(trade, "_trail_done")
                trail_prev = getattr(trade, "_trail_done") if had_trail else _missing
                setattr(trade, "_trail_done", True)
            sig_dict = None
            if signal is not None:
                sig_dict = {"side": signal.side, "strength": signal.strength}
            if is_dict:
                had_price = "current_price" in trade
                prev_price = trade.get("current_price") if had_price else None
                trade["current_price"] = price
            else:
                had_price = hasattr(trade, "current_price")
                prev_price = getattr(trade, "current_price") if had_price else None
                setattr(trade, "current_price", price)
            try:
                decision = rs.manage_position(trade, sig_dict)
            finally:
                if is_dict:
                    if had_price:
                        trade["current_price"] = prev_price
                    else:
                        trade.pop("current_price", None)
                    if had_trail:
                        trade["_trail_done"] = trail_prev
                    else:
                        trade.pop("_trail_done", None)
                    if atr is not None:
                        if atr_prev is _missing:
                            trade.pop("atr", None)
                        else:
                            trade["atr"] = atr_prev
                else:
                    if had_price:
                        setattr(trade, "current_price", prev_price)
                    else:
                        if hasattr(trade, "current_price"):
                            delattr(trade, "current_price")
                    if had_trail:
                        setattr(trade, "_trail_done", trail_prev)
                    else:
                        if hasattr(trade, "_trail_done"):
                            delattr(trade, "_trail_done")
                    if atr is not None:
                        if atr_prev is _missing:
                            if hasattr(trade, "atr"):
                                delattr(trade, "atr")
                        else:
                            setattr(trade, "atr", atr_prev)
            if decision == "close":
                side = "sell" if trade.get("side") == "buy" else "buy"
                close = Signal(side, 1.0)
                close.limit_price = price
                return close
            if decision in {"scale_in", "scale_out"}:
                if signal is not None:
                    return signal
                if symbol is None:
                    return None

                def _to_float(value: Any) -> float | None:
                    try:
                        val = float(value)
                    except (TypeError, ValueError):
                        return None
                    if not math.isfinite(val):
                        return None
                    return val

                target_strength = None
                if isinstance(trade, dict):
                    target_strength = trade.get("strength")
                else:
                    target_strength = getattr(trade, "strength", None)
                target_strength_val = _to_float(target_strength)

                prev_strength_val = None
                targets_map = getattr(rs, "_signal_targets", None)
                if isinstance(targets_map, dict):
                    prev_strength_val = _to_float(targets_map.get(symbol))

                last_sig = getattr(self, "_last_signal", {}).get(symbol)
                if prev_strength_val is None and last_sig is not None:
                    if isinstance(last_sig, dict):
                        prev_strength_val = _to_float(last_sig.get("strength"))
                    else:
                        prev_strength_val = _to_float(getattr(last_sig, "strength", None))

                if target_strength_val is None:
                    return None
                if prev_strength_val is None:
                    prev_strength_val = 0.0 if decision == "scale_in" else target_strength_val

                diff_strength = target_strength_val - prev_strength_val
                tol = 1e-12
                if decision == "scale_out":
                    if diff_strength >= -tol:
                        return None
                    order_strength = abs(diff_strength)
                else:  # scale_in
                    if diff_strength <= tol:
                        return None
                    order_strength = abs(diff_strength)

                trade_side = trade.get("side") if isinstance(trade, dict) else getattr(trade, "side", None)
                trade_side_norm = str(trade_side).lower() if trade_side is not None else None
                if trade_side_norm in {"buy", "long"}:
                    order_side = "buy" if decision == "scale_in" else "sell"
                elif trade_side_norm in {"sell", "short"}:
                    order_side = "sell" if decision == "scale_in" else "buy"
                else:
                    order_side = None
                if order_side is None or order_strength <= tol:
                    return None

                last_limit = None
                last_metadata: dict[str, Any] = {}
                if last_sig is not None:
                    if isinstance(last_sig, dict):
                        last_limit = last_sig.get("limit_price")
                        meta_val = last_sig.get("metadata")
                    else:
                        last_limit = getattr(last_sig, "limit_price", None)
                        meta_val = getattr(last_sig, "metadata", None)
                    if isinstance(meta_val, dict):
                        try:
                            last_metadata = deepcopy(meta_val)
                        except Exception:
                            last_metadata = dict(meta_val)

                synthetic = Signal(order_side, order_strength, reduce_only=(decision == "scale_out"))
                synthetic.limit_price = last_limit if last_limit is not None else price
                if last_metadata:
                    synthetic.metadata.update(last_metadata)
                synthetic.metadata["_target_strength"] = target_strength_val
                if symbol:
                    self._track_requote(key=(symbol, synthetic.side), reset=True)
                signal = synthetic
            if decision == "hold" and signal is not None:
                return signal
            if signal is None:
                return None
        if signal is not None:
            target_override = None
            meta_source: dict[str, Any] | None = None
            if isinstance(signal, dict):
                meta_candidate = signal.get("metadata")
                if isinstance(meta_candidate, dict):
                    meta_source = meta_candidate
            else:
                meta_candidate = getattr(signal, "metadata", None)
                if isinstance(meta_candidate, dict):
                    meta_source = meta_candidate
            if meta_source is not None:
                override_val = meta_source.pop("_target_strength", None)
                try:
                    if override_val is not None:
                        target_override = float(override_val)
                        if not math.isfinite(target_override):
                            target_override = None
                except (TypeError, ValueError):
                    target_override = None
            if symbol:
                if target_override is not None:
                    rs.update_signal_strength(symbol, target_override)
                else:
                    strength_val = (
                        signal["strength"] if isinstance(signal, dict) else signal.strength
                    )
                    rs.update_signal_strength(symbol, strength_val)
            qty = rs.calc_position_size(
                signal["strength"] if isinstance(signal, dict) else signal.strength,
                price,
                volatility=volatility,
                target_volatility=target_volatility,
                clamp=False,
            )
            notional = abs(qty) * price
            if qty < rs.min_order_qty or (
                getattr(rs, "min_notional", 0.0) and notional < rs.min_notional
            ):
                log.info("orden submínima qty=%.8f notional=%.8f", qty, notional)
                return None
        return signal


def load_params(path: str | None) -> dict[str, Any]:
    """Load strategy parameters from a YAML file.

    Parameters
    ----------
    path:
        Location of a YAML file containing a mapping of parameter names to
        values. If ``None`` or empty, an empty dict is returned. The function
        raises :class:`FileNotFoundError` when the path does not exist and
        :class:`ValueError` if the YAML payload is not a mapping.
    """

    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("config file must define a mapping of parameters")
    return data


def record_signal_metrics(liquidity):
    """Decorator factory to time signal generation and persist events.

    Parameters
    ----------
    liquidity:
        Instance of :class:`LiquidityFilterManager` providing the ``passes``
        method used to gate signal generation.
    """

    def decorator(fn):
        def wrapper(self: Strategy, bar: dict[str, Any]) -> Signal | None:  # type: ignore[misc]
            tf = bar.get("timeframe")
            if tf is None:
                tf = getattr(self, "timeframe", None)
                if tf is not None:
                    bar.setdefault("timeframe", tf)
            if not liquidity.passes(bar, tf):
                return None
            start = time.monotonic()
            sig = fn(self, bar)
            gen_ts = time.time()
            if sig is not None:
                sig.signal_ts = gen_ts
            duration = time.monotonic() - start
            REQUEST_LATENCY.labels(method=self.name, endpoint="on_bar").observe(duration)

            # track last signal for re-quoting decisions
            symbol = bar.get("symbol")
            if not hasattr(self, "_last_signal"):
                self._last_signal = {}
            if symbol:
                key = (symbol, bar.get("timeframe"))
                self._last_signal[key] = sig
                self._last_signal[symbol] = sig

            # Risk-based sizing is now handled within ``RiskService.check_order``
            # so strategy signals keep their original strength here.

            if sig and sig.side in {"buy", "sell"} and {"exchange", "symbol"} <= bar.keys():
                try:
                    engine = timescale.get_engine()
                    timescale.insert_order(
                        engine,
                        strategy=self.name,
                        exchange=bar["exchange"],
                        symbol=bar["symbol"],
                        side=sig.side,
                        type_="signal",
                        qty=float(sig.strength),
                        px=bar.get("close"),
                        status="signal",
                        notes=None,
                    )
                except Exception:
                    # Persistence is best-effort only
                    pass
            return sig

        return wrapper

    return decorator


__all__ = [
    "Signal",
    "Strategy",
    "load_params",
    "record_signal_metrics",
    "timeframe_to_minutes",
]
