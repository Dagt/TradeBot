# src/tradingbot/strategies/arbitrage_triangular.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from .base import Strategy, Signal, record_signal_metrics
from ..filters.liquidity import LiquidityFilterManager

PARAM_INFO = {
    "taker_fee_bps": "Comisión taker por tramo en puntos básicos",
    "buffer_bps": "Margen adicional por tramo en puntos básicos",
    "min_edge": "Edge mínimo neto para operar",
}

@dataclass
class TriRoute:
    base: str   # p.ej. "BTC"
    mid: str    # p.ej. "ETH"
    quote: str  # p.ej. "USDT"

@dataclass
class TriSymbols:
    bq: str  # BASE/QUOTE, p.ej. "BTC/USDT"
    mq: str  # MID/QUOTE,  p.ej. "ETH/USDT"
    mb: str  # MID/BASE,   p.ej. "ETH/BTC"

@dataclass
class TriEdge:
    direction: str   # "b->m" (QUOTE->BASE->MID->QUOTE) o "m->b" (QUOTE->MID->BASE->QUOTE)
    gross: float     # edge bruto (sin buffer) sobre 1 QUOTE
    net: float       # edge neto después de buffer_bps
    prices: Dict[str, float]  # {"bq":..., "mq":..., "mb":...}

def make_symbols(route: TriRoute) -> TriSymbols:
    return TriSymbols(
        bq=f"{route.base}/{route.quote}",
        mq=f"{route.mid}/{route.quote}",
        mb=f"{route.mid}/{route.base}",
    )

def compute_edge(
    prices: Dict[str, float],
    taker_fee_bps: float,
    buffer_bps: float,
) -> Optional[TriEdge]:
    """Compute the triangular arbitrage edge.

    The returned :class:`TriEdge` reports both the gross edge (ignoring fees and
    slippage) and the net edge after subtracting ``taker_fee_bps`` and
    ``buffer_bps`` for each leg.  ``prices`` must contain the keys ``bq``
    (BASE/QUOTE), ``mq`` (MID/QUOTE) and ``mb`` (MID/BASE).
    """

    if any(prices.get(k) in (None, 0) for k in ("bq", "mq", "mb")):
        return None

    bq = float(prices["bq"])
    mq = float(prices["mq"])
    mb = float(prices["mb"])

    # Gross edge without fees or buffers
    base_qty_g = 1.0 / bq
    mid_qty_g = base_qty_g / mb
    quote_out_bm_g = mid_qty_g * mq
    edge_bm_g = quote_out_bm_g - 1.0

    mid_qty2_g = 1.0 / mq
    base_qty2_g = mid_qty2_g / mb
    quote_out_mb_g = base_qty2_g * bq
    edge_mb_g = quote_out_mb_g - 1.0

    # Net edge applying fees and buffer to each leg
    f = 1 - taker_fee_bps / 10000.0
    buf = 1 - buffer_bps / 10000.0

    base_qty = (1.0 * f * buf) / bq
    mid_qty = (base_qty * f * buf) / mb
    quote_out_bm = mid_qty * mq * f * buf
    edge_bm = quote_out_bm - 1.0

    mid_qty2 = (1.0 * f * buf) / mq
    base_qty2 = (mid_qty2 * f * buf) / mb
    quote_out_mb = base_qty2 * bq * f * buf
    edge_mb = quote_out_mb - 1.0

    if edge_bm > edge_mb:
        return TriEdge(direction="b->m", gross=edge_bm_g, net=edge_bm, prices=prices)
    else:
        return TriEdge(direction="m->b", gross=edge_mb_g, net=edge_mb, prices=prices)

liquidity = LiquidityFilterManager()


class TriangularArb(Strategy):
    """Naive triangular arbitrage strategy based on three market prices.

    Parameters
    ----------
    taker_fee_bps:
        Taker fee in basis points applied to each leg.
    buffer_bps:
        Additional buffer in basis points applied to each leg.
    min_edge:
        Minimum net edge required to emit a trading signal.

    The strategy expects the incoming ``bar`` to contain a ``prices`` mapping
    with the keys ``bq`` (base/quote), ``mq`` (mid/quote) and ``mb`` (mid/base).
    A ``buy`` signal represents traversing the markets in the ``b->m``
    direction, while ``sell`` corresponds to ``m->b``.
    """

    name = "triangular_arb"

    def __init__(self, **kwargs):
        self.taker_fee_bps = kwargs.get("taker_fee_bps", 0.0)
        self.buffer_bps = kwargs.get("buffer_bps", 0.0)
        self.min_edge = kwargs.get("min_edge", 0.0)
        # Accept optional ``risk_service`` for interface compatibility.
        self.risk_service = kwargs.get("risk_service")

    @record_signal_metrics(liquidity)
    def on_bar(self, bar: Dict[str, Dict[str, float]]) -> Optional[Signal]:
        prices = bar.get("prices") if isinstance(bar, dict) else None
        if not prices:
            return None
        edge = compute_edge(prices, self.taker_fee_bps, self.buffer_bps)
        if edge and edge.net > self.min_edge:
            strength = max(0.0, edge.net)
            if strength > 0:
                side = "buy" if edge.direction == "b->m" else "sell"
                return Signal(side, strength)
        return Signal("flat", 0.0)
