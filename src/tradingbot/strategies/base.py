from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    signal_ts: float | None = None

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

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...

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
        return "re_quote" if self._edge_still_exists(order) else None

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

        atr_map = getattr(self, "_last_atr", {})
        atr_val = atr_map.get(order.symbol)
        if atr_val:
            offset = 0.1 * atr_val
            price = order.price if order.price is not None else res.get("price")
            if price is None:
                price = 0.0
            if order.side == "buy":
                price += offset
            else:
                price -= offset
            order.price = price
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

        if signal is not None and signal.limit_price is None:
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

        symbol = bar.get("symbol")
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
            if decision in {"scale_in", "scale_out"} and signal is not None:
                return signal
            if decision == "hold" and signal is not None:
                return signal
            return None
        if signal is not None:
            qty = rs.calc_position_size(
                signal.strength,
                price,
                volatility=volatility,
                target_volatility=target_volatility,
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
