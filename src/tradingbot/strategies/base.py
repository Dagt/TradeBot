from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import time

import yaml

from ..utils.metrics import REQUEST_LATENCY
from ..storage import timescale

@dataclass
class Signal:
    side: str  # 'buy' | 'sell' | 'flat'
    strength: float = 1.0
    reduce_only: bool = False

class Strategy(ABC):
    name: str

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...


class StatefulStrategy(Strategy):
    """Strategy base class with simple position tracking and exits."""

    def __init__(
        self,
        *,
        tp_pct: float = 0.0,
        sl_pct: float = 0.0,
        max_hold_bars: int = 0,
    ) -> None:
        self.tp_pct = float(tp_pct)
        self.sl_pct = float(sl_pct)
        self.max_hold_bars = int(max_hold_bars)
        self.pos_side: str | None = None
        self.entry_price: float | None = None
        self.hold_bars: int = 0

    def check_exit(self, price: float) -> Signal | None:
        """Check exit conditions and return an opposing signal if met."""

        if self.pos_side is None or self.entry_price is None:
            return None
        self.hold_bars += 1
        if self.pos_side == "buy":
            hit_tp = self.tp_pct > 0 and price >= self.entry_price * (1 + self.tp_pct)
            hit_sl = self.sl_pct > 0 and price <= self.entry_price * (1 - self.sl_pct)
            timed_out = self.max_hold_bars > 0 and self.hold_bars >= self.max_hold_bars
            if hit_tp or hit_sl or timed_out:
                self.pos_side = None
                self.entry_price = None
                self.hold_bars = 0
                return Signal("sell", 1.0)
        else:
            hit_tp = self.tp_pct > 0 and price <= self.entry_price * (1 - self.tp_pct)
            hit_sl = self.sl_pct > 0 and price >= self.entry_price * (1 + self.sl_pct)
            timed_out = self.max_hold_bars > 0 and self.hold_bars >= self.max_hold_bars
            if hit_tp or hit_sl or timed_out:
                self.pos_side = None
                self.entry_price = None
                self.hold_bars = 0
                return Signal("buy", 1.0)
        return None

    def open_position(self, side: str, price: float) -> Signal:
        self.pos_side = side
        self.entry_price = price
        self.hold_bars = 0
        return Signal(side, 1.0)


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
        start = time.monotonic()
        sig = fn(self, bar)
        duration = time.monotonic() - start
        REQUEST_LATENCY.labels(method=self.name, endpoint="on_bar").observe(duration)

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
                )
            except Exception:
                # Persistence is best-effort only
                pass
        return sig

    return wrapper


__all__ = [
    "Signal",
    "Strategy",
    "StatefulStrategy",
    "load_params",
    "record_signal_metrics",
]
