from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import yaml

from ..utils.metrics import REQUEST_LATENCY
from ..storage import timescale
from ..filters import passes as liquidity_passes
from ..execution.order_types import Order

@dataclass
class Signal:
    """Simple trading signal.

    ``strength`` defines sizing through ``notional = equity * strength``. A
    value of ``1.0`` uses the full account equity, values above ``1.0`` pyramid
    exposure and values between ``0`` and ``1`` scale it down. ``0`` or
    negative values close the position. Optional ``take_profit`` and
    ``stop_loss`` levels express absolute prices for OCO orders. ``risk_pct``
    acts as a local stop‑loss calculated as ``notional * risk_pct``.
    """

    side: str  # 'buy' | 'sell' | 'flat'
    strength: float = 1.0  # fraction of the base equity allocation
    reduce_only: bool = False
    take_profit: float | None = None  # optional TP price level
    stop_loss: float | None = None  # optional SL price level
    trailing_pct: float | None = None  # optional trailing stop distance as fraction
    expected_edge_bps: float | None = None  # optional expected edge in basis points

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
        direction.
        """

        if not hasattr(self, "pending_qty"):
            self.pending_qty: dict[str, float] = {}
        pending = float(res.get("pending_qty", order.qty))
        self.pending_qty[order.symbol] = pending
        if pending <= 0:
            return None
        return "re_quote" if self._edge_still_exists(order) else None


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


def record_signal_metrics(fn):
    """Decorator to time strategy signal generation and persist events.

    It observes ``REQUEST_LATENCY`` for each ``on_bar`` invocation and persists
    generated signals as orders into Timescale if available. Any persistence
    errors are silently ignored so strategies remain side-effect free in test
    environments.
    """

    def wrapper(self: Strategy, bar: dict[str, Any]) -> Signal | None:  # type: ignore[misc]
        if not liquidity_passes(bar):
            return None
        start = time.monotonic()
        sig = fn(self, bar)
        duration = time.monotonic() - start
        REQUEST_LATENCY.labels(method=self.name, endpoint="on_bar").observe(duration)

        # track last signal for re-quoting decisions
        symbol = bar.get("symbol")
        if not hasattr(self, "_last_signal"):
            self._last_signal = {}
        if symbol:
            self._last_signal[symbol] = sig

        # Risk-based sizing is now handled within ``RiskService.check_order``
        # so strategy signals keep their original strength here.

        if sig and sig.side in {"buy", "sell"} and {"exchange", "symbol"} <= bar.keys():
            try:
                engine = timescale.get_engine()
                notes = None
                if sig.expected_edge_bps is not None:
                    notes = {"expected_edge_bps": float(sig.expected_edge_bps)}
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
                    notes=notes,
                )
            except Exception:
                # Persistence is best-effort only
                pass
        return sig

    return wrapper


__all__ = ["Signal", "Strategy", "load_params", "record_signal_metrics"]
