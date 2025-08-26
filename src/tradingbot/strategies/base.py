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
    """Simple trading signal.

    ``strength`` defines sizing through ``notional = equity * strength``. A
    value of ``1.0`` uses the full account equity, values above ``1.0`` pyramid
    exposure and values between ``0`` and ``1`` scale it down. ``0`` or
    negative values close the position. ``risk_pct`` acts as a local stop‑loss
    calculated as ``notional * risk_pct``.
    """

    side: str  # 'buy' | 'sell' | 'flat'
    strength: float = 1.0  # fraction of the base equity allocation
    reduce_only: bool = False

class Strategy(ABC):
    name: str

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...


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


__all__ = ["Signal", "Strategy", "load_params", "record_signal_metrics"]
