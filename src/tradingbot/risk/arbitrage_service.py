from __future__ import annotations

"""Simple risk utilities dedicated to arbitrage strategies.

The generic :mod:`tradingbot.risk.service` module contains a rather heavyweight
risk manager tailored for directional strategies.  The arbitrage strategies in
this repository only need a lightweight component that validates whether an
opportunity remains profitable after accounting for fees and slippage and that
keeps track of a few exposure caps.  This module purposely implements just that
subset so triangular and cross‑exchange arbitrage can operate without pulling in
all the machinery from :class:`RiskService`.
"""

from dataclasses import dataclass, field
import asyncio
from collections import defaultdict
from typing import Dict, Tuple


@dataclass
class ArbGuardConfig:
    """Configuration for :class:`ArbitrageRiskService`.

    Parameters
    ----------
    max_position_usd:
        Upper bound for the notional allocated to a single trade.
    max_open_trades:
        Maximum number of simultaneous open arbitrage trades.
    min_edge:
        Minimum net edge required to deploy capital.
    slippage_buffer_bps:
        Additional buffer to subtract from the edge to account for
        unexpected slippage.  Expressed in basis points.
    """

    max_position_usd: float = float("inf")
    max_open_trades: int = 1
    min_edge: float = 0.0
    slippage_buffer_bps: float = 0.0


class ArbitrageRiskService:
    """Lightweight risk helper for arbitrage strategies.

    The service exposes a minimal async API that guards capital allocation and
    keeps track of open exposures.  It deliberately avoids the complexity of the
    generic risk engine while still providing basic limits and validations that
    are relevant for arbitrage where trades are expected to be hedged quickly.
    """

    def __init__(self, cfg: ArbGuardConfig | None = None) -> None:
        self.cfg = cfg or ArbGuardConfig()
        self._open_trades = 0
        self._lock = asyncio.Lock()
        self._positions: Dict[str, float] = defaultdict(float)

    async def evaluate(
        self,
        symbol: str,
        edge: float,
        price_ref: float,
        account_equity: float,
        *,
        fees: float = 0.0,
        slippage: float = 0.0,
    ) -> Tuple[float, float]:
        """Validate an opportunity and return the allowed notional.

        Parameters
        ----------
        symbol:
            Market symbol being evaluated.
        edge:
            Raw edge expressed as a decimal (e.g. ``0.001`` for 10 bps).
        price_ref:
            Reference price used to translate notional into quantity.
        account_equity:
            Equity available for arbitrage expressed in quote currency.
        fees:
            Sum of fees across all legs expressed as a decimal.
        slippage:
            Expected slippage across all legs expressed as a decimal.

        Returns
        -------
        tuple
            ``(notional, net_edge)`` where ``notional`` is the capital to
            allocate (0 if the trade is rejected) and ``net_edge`` is the edge
            after deductions.
        """

        async with self._lock:
            net_edge = edge - fees - slippage - self.cfg.slippage_buffer_bps / 10000.0
            if net_edge <= self.cfg.min_edge:
                return 0.0, net_edge
            if self._open_trades >= self.cfg.max_open_trades:
                return 0.0, net_edge
            notional = min(self.cfg.max_position_usd, account_equity * abs(net_edge))
            if notional <= 0:
                return 0.0, net_edge
            self._open_trades += 1
            return notional, net_edge

    async def on_fill(self, symbol: str, side: str, qty: float) -> None:
        """Record a fill and release an open trade slot."""
        async with self._lock:
            signed = float(qty) if side == "buy" else -float(qty)
            self._positions[symbol] += signed
            if self._open_trades > 0:
                self._open_trades -= 1

    def exposure(self, symbol: str) -> float:
        """Return current net position for ``symbol``."""
        return self._positions.get(symbol, 0.0)


__all__ = ["ArbitrageRiskService", "ArbGuardConfig"]
