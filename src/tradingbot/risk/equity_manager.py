from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EquityRiskManager:
    """Simple equity-based risk manager.

    Tracks peak equity and per-position PnL to allow dynamic resizing of
    positions.  Stop-loss and trailing stop limits are expressed as
    percentages of account equity rather than price levels.
    """

    equity: float
    stop_loss_pct: float = 0.0
    trailing_stop_pct: float = 0.0

    peak_equity: float = field(init=False)
    pnl_positions: Dict[str, float] = field(default_factory=dict, init=False)
    _pnl_peaks: Dict[str, float] = field(default_factory=dict, init=False)
    _signals: Dict[str, float] = field(default_factory=dict, init=False)
    _positions: Dict[str, float] = field(default_factory=dict, init=False)
    enabled: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        self.equity = float(self.equity)
        self.stop_loss_pct = abs(self.stop_loss_pct)
        self.trailing_stop_pct = abs(self.trailing_stop_pct)
        self.peak_equity = self.equity
        self.entry_equity = self.equity

    # ------------------------------------------------------------------
    # Tracking helpers
    def update_equity(self, equity: float) -> None:
        """Update current equity and peak."""
        self.equity = float(equity)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

    def update_position_pnl(self, symbol: str, pnl: float) -> None:
        """Store latest PnL for ``symbol`` and keep peak PnL."""
        self.pnl_positions[symbol] = float(pnl)
        peak = self._pnl_peaks.get(symbol)
        if peak is None or pnl > peak:
            self._pnl_peaks[symbol] = float(pnl)

    # ------------------------------------------------------------------
    # Position sizing helpers
    def increase_position(
        self,
        symbol: str,
        qty: float,
        signal: float,
        pnl_positive_threshold: float,
        increment_pct: float = 0.5,
    ) -> float:
        """Increase ``qty`` if ``signal`` grows or PnL exceeds threshold."""
        prev_signal = self._signals.get(symbol, float("-inf"))
        pnl = self.pnl_positions.get(symbol, 0.0)
        new_qty = qty
        if signal > prev_signal or pnl > pnl_positive_threshold:
            new_qty = qty * (1 + increment_pct)
        self._signals[symbol] = signal
        self._positions[symbol] = new_qty
        return new_qty

    def reduce_position_on_trailing_stop(self, symbol: str, qty: float) -> float:
        """Reduce position size if PnL retraces by ``trailing_stop_pct``."""
        pnl = self.pnl_positions.get(symbol, 0.0)
        peak = self._pnl_peaks.get(symbol, pnl)
        if peak > 0:
            retracement = (peak - pnl) / peak
            if self.trailing_stop_pct > 0 and retracement >= self.trailing_stop_pct:
                new_qty = qty * 0.5
                self._positions[symbol] = new_qty
                return new_qty
        self._positions[symbol] = qty
        return qty

    # ------------------------------------------------------------------
    # Limit checks
    def check_limits(self, equity: float) -> bool:
        """Check stop-loss and trailing stop based on ``equity``."""
        if not self.enabled:
            return False
        self.update_equity(equity)
        loss_pct = 0.0
        if self.entry_equity > 0:
            loss_pct = (self.entry_equity - self.equity) / self.entry_equity
        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
        if self.stop_loss_pct > 0 and loss_pct >= self.stop_loss_pct:
            self.enabled = False
            return False
        if self.trailing_stop_pct > 0 and drawdown >= self.trailing_stop_pct:
            self.enabled = False
            return False
        return True
