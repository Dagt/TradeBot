"""Monitoring helpers."""
from dataclasses import dataclass


@dataclass
class Monitor:
    """Collect and report runtime metrics."""

    def emit(self, name: str, value: float) -> None:
        """Emit a metric (placeholder)."""
        print(f"METRIC {name}={value}")
