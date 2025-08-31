# src/tradingbot/execution/balance.py
from __future__ import annotations
from dataclasses import dataclass

from typing import Dict, Mapping, Any

from ..risk.service import RiskService
from ..storage.timescale import insert_portfolio_snapshot

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


async def rebalance_between_exchanges(
    asset: str,
    price: float,
    venues: Mapping[str, Any],
    risk: RiskService,
    engine,
    *,
    threshold: float = 0.0,
) -> None:
    """Rebalance ``asset`` holdings across ``venues``.

    Parameters
    ----------
    asset: str
        Asset/currency to rebalance, e.g. ``"USDT"``.
    price: float
        Reference price in USD for snapshot notional calculation.
    venues: Mapping[str, Any]
        Mapping of venue name to a connector exposing ``fetch_balance`` and
        ``transfer`` methods (synchronous).
    risk: RiskService
        Risk service instance whose positions will be synchronised.
    engine: Any
        SQLAlchemy engine for persistence through ``insert_portfolio_snapshot``.
    threshold: float, optional
        Minimum balance difference required to trigger a transfer.
    """

    # Fetch balances for all venues
    balances: Dict[str, float] = {}
    for name, client in venues.items():
        bal = await __to_thread(client.fetch_balance)
        balances[name] = float((bal.get(asset) or {}).get("free") or 0.0)

    if len(balances) < 2:
        return

    max_venue = max(balances, key=balances.get)
    min_venue = min(balances, key=balances.get)
    diff = balances[max_venue] - balances[min_venue]
    if diff <= threshold:
        # No action required
        for venue, bal in balances.items():
            risk.update_position(venue, asset, bal)
        return

    amount = diff / 2
    src = venues[max_venue]
    dst = venues[min_venue]
    # Execute transfer using blocking call in thread
    await __to_thread(src.transfer, asset, amount, dst)

    balances[max_venue] -= amount
    balances[min_venue] += amount

    # Update risk manager and persist snapshots
    for venue, bal in balances.items():
        risk.update_position(venue, asset, bal)
        insert_portfolio_snapshot(
            engine,
            venue=venue,
            symbol=asset,
            position=bal,
            price=price,
            notional_usd=bal * price,
        )

