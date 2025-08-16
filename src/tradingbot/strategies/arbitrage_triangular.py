# src/tradingbot/strategies/arbitrage_triangular.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from .base import Strategy, Signal

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

def compute_edge(prices: Dict[str, float], taker_fee_bps: float, buffer_bps: float) -> Optional[TriEdge]:
    """
    prices: {"bq": Px(BASE/QUOTE), "mq": Px(MID/QUOTE), "mb": Px(MID/BASE)}
    taker_fee_bps: comisión de taker por pata (en bps). Ejem: 7.5 bps = 0.075%
    buffer_bps: colchón para slippage/errores (por pata) en bps.
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

def compute_qtys_for_route(
    direction: str,
    notional_quote: float,
    prices: Dict[str, float],
    taker_fee_bps: float,
    buffer_bps: float,
) -> Dict[str, float]:
    """
    Devuelve cantidades por pata (aprox) dadas las prices y un notional en QUOTE.
    Todas las patas se asumen taker a market (simulación).
    """
    f = 1 - taker_fee_bps/10000.0
    buf = 1 - buffer_bps/10000.0
    bq, mq, mb = prices["bq"], prices["mq"], prices["mb"]

    if direction == "b->m":
        # QUOTE -> BASE (BUY BASE/QUOTE)
        base_qty = (notional_quote * f * buf) / bq
        # BASE -> MID (BUY MID/BASE)
        mid_qty = (base_qty * f * buf) / mb
        # MID -> QUOTE (SELL MID/QUOTE)
        quote_out = mid_qty * mq * f * buf
        return {"base_qty": base_qty, "mid_qty": mid_qty, "quote_out": quote_out}
    else:
        # QUOTE -> MID (BUY MID/QUOTE)
        mid_qty = (notional_quote * f * buf) / mq
        # MID -> BASE (SELL MID/BASE) => recibes BASE
        base_qty = (mid_qty * f * buf) / mb
        # BASE -> QUOTE (SELL BASE/QUOTE)
        quote_out = base_qty * bq * f * buf
        return {"base_qty": base_qty, "mid_qty": mid_qty, "quote_out": quote_out}


class TriangularArb(Strategy):
    """Naive triangular arbitrage strategy based on three market prices.

    The strategy expects the incoming ``bar`` to contain a ``prices`` mapping
    with the keys ``bq`` (base/quote), ``mq`` (mid/quote) and ``mb`` (mid/base).
    ``taker_fee_bps`` and ``buffer_bps`` can be supplied via ``**kwargs`` to
    account for trading fees and a safety buffer.

    A ``buy`` signal represents traversing the markets in the ``b->m``
    direction, while ``sell`` corresponds to ``m->b``.
    """

    name = "triangular_arb"

    def __init__(self, **kwargs):
        self.taker_fee_bps = kwargs.get("taker_fee_bps", 0.0)
        self.buffer_bps = kwargs.get("buffer_bps", 0.0)
        self.min_edge = kwargs.get("min_edge", 0.0)

    def on_bar(self, bar: Dict[str, Dict[str, float]]) -> Optional[Signal]:
        prices = bar.get("prices") if isinstance(bar, dict) else None
        if not prices:
            return None
        edge = compute_edge(prices, self.taker_fee_bps, self.buffer_bps)
        if edge and edge.net > self.min_edge:
            side = "buy" if edge.direction == "b->m" else "sell"
            return Signal(side, edge.net)
        return Signal("flat", 0.0)
