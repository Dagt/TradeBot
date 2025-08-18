# src/tradingbot/risk/pnl.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Position:
    qty: float = 0.0            # base units; spot >=0; futures +/- (long>0 / short<0)
    avg_price: float = 0.0      # USDT/base
    realized_pnl: float = 0.0   # acumulado (USDT)
    fees_paid: float = 0.0      # acumulado (USDT)

def _fee_usdt(px: float, qty: float, fee_bps: float) -> float:
    return abs(px * qty) * (fee_bps / 10000.0)

def apply_fill(pos: Position, side: str, qty: float, px: float, fee_bps: float, venue_type: str) -> Position:
    """
    Actualiza la posición con un fill estimado:
      - SPOT: qty >=0, solo lados buy (incrementa) y sell (reduce). No permitimos qty final <0.
      - FUTURES: qty con signo implícito por side (buy aumenta, sell reduce), puede cruzar 0 y voltear lado.
    Realiza PnL cuando se reduce en contra del lado actual.
    """
    q = float(qty)
    px = float(px)
    side = side.lower()
    fee = _fee_usdt(px, q, fee_bps)

    if venue_type == "spot":
      # Representamos spot como qty>=0 siempre
      if pos.qty == 0:
          if side == "buy":
              pos.avg_price = px
              pos.qty = q
          else:
              # vender sin inventario => ignora
              return pos
      else:
          if side == "buy":
              new_qty = pos.qty + q
              pos.avg_price = (pos.avg_price * pos.qty + px * q) / max(new_qty, 1e-12)
              pos.qty = new_qty
          else:  # sell
              reduce_qty = min(q, pos.qty)
              # Realized sobre lo reducido
              pos.realized_pnl += (px - pos.avg_price) * reduce_qty
              pos.qty -= reduce_qty
              # si vende más de la posición, truncamos a 0 (no short en spot)
              if pos.qty < 1e-12:
                  pos.qty = 0.0
      pos.fees_paid += fee
      return pos

    # FUTURES (qty con signo a través de side)
    dir_sign = 1.0 if side == "buy" else -1.0
    incoming = dir_sign * q
    # Si misma dirección que la posición actual (o sin posición): promedio
    if pos.qty == 0 or (pos.qty > 0 and incoming > 0) or (pos.qty < 0 and incoming < 0):
        new_qty = pos.qty + incoming
        # promedio ponderado por notional absoluto
        pos.avg_price = (abs(pos.avg_price * pos.qty) + abs(px * incoming)) / max(abs(new_qty), 1e-12)
        pos.qty = new_qty
    else:
        # Reduce o potencialmente revierte
        if abs(incoming) <= abs(pos.qty):
            # reduce parcialmente
            realized = (pos.avg_price - px) * incoming  # cuidado con signo: incoming <0 (sell) o >0 (buy)
            # Para interpretar correctamente:
            # Si está long (qty>0) y vendemos (incoming<0): realized = (px - avg)*|incoming|
            # Si está short (qty<0) y compramos (incoming>0): realized = (avg - px)*|incoming|
            if pos.qty > 0 and incoming < 0:
                realized = (px - pos.avg_price) * abs(incoming)
            elif pos.qty < 0 and incoming > 0:
                realized = (pos.avg_price - px) * abs(incoming)
            pos.realized_pnl += realized
            pos.qty += incoming
            if abs(pos.qty) < 1e-12:
                pos.qty = 0.0
        else:
            # cruza a lado contrario: cierra todo y re‑abre el exceso al nuevo precio
            close_qty = abs(pos.qty)
            if pos.qty > 0 and incoming < 0:
                realized = (px - pos.avg_price) * close_qty
            else:  # pos.qty < 0 and incoming > 0
                realized = (pos.avg_price - px) * close_qty
            pos.realized_pnl += realized
            # nueva posición con el remanente
            rem = abs(incoming) - close_qty
            pos.qty = (1.0 if incoming > 0 else -1.0) * rem
            pos.avg_price = px
    pos.fees_paid += fee
    return pos

def mark_to_market(pos: Position, mark_px: float) -> float:
    """ UPnL: (mark - avg)*qty (con signo de qty). En spot qty>=0; en futures qty puede ser +/- """
    return (float(mark_px) - pos.avg_price) * pos.qty

def position_from_db(row: dict | None) -> Position:
    """Crea Position desde fila de BD (o posición vacía si None)."""
    if not row:
        return Position()
    return Position(
        qty=float(row.get("qty", 0.0) or 0.0),
        avg_price=float(row.get("avg_price", 0.0) or 0.0),
        realized_pnl=float(row.get("realized_pnl", 0.0) or 0.0),
        fees_paid=float(row.get("fees_paid", 0.0) or 0.0),
    )

