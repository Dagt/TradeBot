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

    if rules.min_qty and qty < rules.min_qty:
        # aumenta al mínimo permitido (al piso del grid)
        if rules.qty_step:
            # subimos al múltiplo más cercano >= min_qty
            mult = int((rules.min_qty + rules.qty_step - 1e-12) / rules.qty_step)
            qty = round(mult * rules.qty_step, 12)
        else:
            qty = rules.min_qty

    # notional para validar (usa mark_price si market)
    px = price if price is not None else mark_price
    notional = (px or 0.0) * qty

    if rules.min_notional and notional < rules.min_notional:
        # escalar qty al mínimo notional
        target_qty = (rules.min_notional / (px or 1.0))
        if rules.qty_step:
            mult = int((target_qty + rules.qty_step - 1e-12) / rules.qty_step)
            target_qty = round(mult * rules.qty_step, 12)
        qty = max(qty, target_qty)
        notional = (px or 0.0) * qty

    # Validación final básica
    if qty <= 0:
        return AdjustResult(price=price, qty=qty, notional=notional, ok=False, reason="qty<=0 tras ajuste")
    if rules.min_notional and notional < rules.min_notional:
        return AdjustResult(price=price, qty=qty, notional=notional, ok=False, reason="minNotional no alcanzado")
    return AdjustResult(price=price, qty=qty, notional=notional, ok=True)
