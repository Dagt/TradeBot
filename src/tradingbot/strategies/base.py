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

class Strategy(ABC):
    name: str

    @abstractmethod
    def on_bar(self, bar: dict[str, Any]) -> Signal | None:
        ...


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


def load_params(path: str) -> dict[str, Any]:
    """Load strategy parameters from a YAML file.

    Parameters
    ----------
    path : str
        Path to a YAML file containing a mapping of parameter names to
        values.

    Returns
    -------
    dict[str, Any]
        Dictionary with parameters. Returns an empty dict if the file is not
        found or cannot be parsed.
    """

    try:
        with Path(path).open("r") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


__all__ = ["Signal", "Strategy", "record_signal_metrics", "load_params"]
