# src/tradingbot/strategies/arbitrage_triangular.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from .base import Strategy, Signal, record_signal_metrics

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

    Parameters
    ----------
    prices:
        Mapping with keys ``"bq"`` (BASE/QUOTE), ``"mq"`` (MID/QUOTE) and
        ``"mb"`` (MID/BASE).
    taker_fee_bps:
        Commission of taker per leg in basis points (e.g. ``7.5`` → 0.075%).
    buffer_bps:
        Extra buffer to account for slippage per leg in basis points.
    """
    if any(prices.get(k) in (None, 0) for k in ("bq", "mq", "mb")):
        return None

    f = 1 - taker_fee_bps/10000.0
    buf = 1 - buffer_bps/10000.0

    bq = float(prices["bq"])
    mq = float(prices["mq"])
    mb = float(prices["mb"])

    # Ruta 1 ("b->m"): QUOTE -> BASE -> MID -> QUOTE
    # 1) Comprar BASE con QUOTE a bq
    base_qty = (1.0 * f * buf) / bq
    # 2) Comprar MID con BASE a mb (precio MID/BASE)
    mid_qty = (base_qty * f * buf) / mb
    # 3) Vender MID por QUOTE a mq
    quote_out_bm = mid_qty * mq * f * buf
    edge_bm = quote_out_bm - 1.0

    # Ruta 2 ("m->b"): QUOTE -> MID -> BASE -> QUOTE
    # 1) Comprar MID con QUOTE a mq
    mid_qty2 = (1.0 * f * buf) / mq
    # 2) Vender MID por BASE a mb (recibes BASE = MID / mb)
    base_qty2 = (mid_qty2 * f * buf) / mb
    # 3) Vender BASE por QUOTE a bq
    quote_out_mb = base_qty2 * bq * f * buf
    edge_mb = quote_out_mb - 1.0

    if edge_bm > edge_mb:
        return TriEdge(direction="b->m", gross=edge_bm, net=edge_bm, prices=prices)
    else:
        return TriEdge(direction="m->b", gross=edge_mb, net=edge_mb, prices=prices)

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

    @record_signal_metrics
    def on_bar(self, bar: Dict[str, Dict[str, float]]) -> Optional[Signal]:
        prices = bar.get("prices") if isinstance(bar, dict) else None
        if not prices:
            return None
        edge = compute_edge(prices, self.taker_fee_bps, self.buffer_bps)
        if edge and edge.net > self.min_edge:
            strength = max(0.0, min(edge.net, 1.0))
            if strength > 0:
                side = "buy" if edge.direction == "b->m" else "sell"
                return Signal(side, strength)
        return Signal("flat", 0.0)
