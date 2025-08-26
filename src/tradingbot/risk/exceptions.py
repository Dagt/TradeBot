"""Standard risk-related exceptions."""

from __future__ import annotations


class StopLossExceeded(Exception):
    """Signal that only the affected position should be closed.

    Raised when unrealized losses breach the configured ``risk_pct``.
    Unlike a kill switch, this indicates the system should remain enabled
    while the corresponding position is exited.
    """


__all__ = ["StopLossExceeded"]

