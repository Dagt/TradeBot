# src/tradingbot/execution/normalize.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

def _round_step(x: float, step: float) -> float:
    if step is None or step == 0:
        return x
    return round(round(x / step) * step, 12)


def _floor_step(x: float, step: float) -> float:
    if step is None or step == 0:
        return x
    n = int(x / step)
    return round(n * step, 12)


def floor_to_step(x: float, step: float) -> float:
    """Floor ``x`` to the nearest multiple of ``step`` preserving sign."""

    return _floor_step(x, step)

@dataclass
class SymbolRules:
    price_step: Optional[float] = None   # tickSize
    qty_step: Optional[float] = None     # stepSize
    min_qty: Optional[float] = None      # minQty
    min_notional: Optional[float] = None # minNotional

@dataclass
class AdjustResult:
    price: Optional[float]
    qty: float
    notional: float
    ok: bool
    reason: Optional[str] = None

def adjust_order(price: Optional[float], qty: float, mark_price: float, rules: SymbolRules, side: str) -> AdjustResult:
    """
    Ajusta price/qty al grid permitido y valida minNotional.
    - MARKET: price=None -> se usa mark_price solo para validar notional.
    - LIMIT:  price se alinea a tickSize.
    Qty siempre se alinea a stepSize y respeta minQty.
    """
    # redondeos
    if price is not None and rules.price_step:
        price = _round_step(price, rules.price_step)

    if rules.qty_step:
        qty = _floor_step(qty, rules.qty_step)

    # notional para validar (usa mark_price si market)
    px = price if price is not None else mark_price
    notional = (px or 0.0) * qty

    if qty <= 0:
        return AdjustResult(
            price=price,
            qty=qty,
            notional=notional,
            ok=False,
            reason="qty<=0 tras ajuste",
        )

    if rules.min_qty and qty < rules.min_qty:
        return AdjustResult(
            price=price,
            qty=qty,
            notional=notional,
            ok=False,
            reason="below_min_qty",
        )

    if rules.min_notional and notional < rules.min_notional:
        return AdjustResult(
            price=price,
            qty=qty,
            notional=notional,
            ok=False,
            reason="below_min_notional",
        )

    return AdjustResult(price=price, qty=qty, notional=notional, ok=True)
