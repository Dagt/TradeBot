# src/tradingbot/execution/balance.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SpotBalances:
    free_base: float
    free_quote: float

async def fetch_spot_balances(ccxt_client, base: str, quote: str) -> SpotBalances:
    bal = await __to_thread(ccxt_client.fetch_balance)
    fb = float((bal.get(base) or {}).get("free") or 0.0)
    fq = float((bal.get(quote) or {}).get("free") or 0.0)
    return SpotBalances(free_base=fb, free_quote=fq)

async def __to_thread(fn, *args, **kwargs):
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)

def cap_qty_by_balance_spot(side: str, qty: float, price: float, balances: SpotBalances, utilization: float = 0.95) -> float:
    """
    BUY: limita por free_quote -> max_qty = (free_quote * util) / price
    SELL: limita por free_base  -> max_qty = free_base * util
    """
    if side.lower() == "buy":
        max_qty = (balances.free_quote * utilization) / max(price, 1e-12)
    else:
        max_qty = balances.free_base * utilization
    return max(0.0, min(qty, max_qty))

@dataclass
class FuturesBalances:
    free_usdt: float

async def fetch_futures_usdt_free(ccxt_client) -> FuturesBalances:
    bal = await __to_thread(ccxt_client.fetch_balance, params={"type": "future"})
    # CCXT suele exponer 'USDT' para USDⓈ-M
    free_usdt = float((bal.get("USDT") or {}).get("free") or 0.0)
    return FuturesBalances(free_usdt=free_usdt)

def cap_qty_by_balance_futures(side: str, qty: float, price: float, free_usdt: float, leverage: int, utilization: float = 0.5) -> float:
    """
    Aprox: notional permitido ≈ free_usdt * leverage * utilization
    max_qty = notional / price
    """
    notional_cap = max(0.0, free_usdt * leverage * utilization)
    max_qty = notional_cap / max(price, 1e-12)
    return max(0.0, min(qty, max_qty))
