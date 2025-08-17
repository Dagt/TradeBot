from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from ..adapters.base import ExchangeAdapter
from ..adapters.deribit import DeribitAdapter
from ..adapters.deribit_ws import DeribitWSAdapter

try:
    from ..storage.timescale import get_engine, insert_cross_signal, insert_fill
    _CAN_PG = True
except Exception:  # pragma: no cover - Timescale optional
    _CAN_PG = False

log = logging.getLogger(__name__)


@dataclass
class CrossArbConfig:
    """Configuration for cross exchange spot/perp arbitrage."""

    symbol: str
    spot: ExchangeAdapter
    perp: ExchangeAdapter
    threshold: float = 0.001  # premium threshold as decimal (0.001 = 0.1%)
    notional: float = 100.0  # quote currency notional per leg
    persist_pg: bool = False


async def run_cross_exchange_arbitrage(cfg: CrossArbConfig) -> None:
    """Run a simple spot/perp cross exchange arbitrage loop.

    The function listens to trade streams from both adapters. When the
    difference between perp and spot price exceeds ``cfg.threshold`` it sends
    offsetting market orders on each venue and persists the opportunity and
    resulting fills into TimescaleDB (if available).
    """
    if isinstance(cfg.spot, DeribitAdapter):
        cfg.spot = DeribitWSAdapter(rest=cfg.spot)
    if isinstance(cfg.perp, DeribitAdapter):
        cfg.perp = DeribitWSAdapter(rest=cfg.perp)

    last: Dict[str, Optional[float]] = {"spot": None, "perp": None}
    balances: Dict[str, float] = {cfg.spot.name: 0.0, cfg.perp.name: 0.0}
    trade_lock = asyncio.Lock()

    engine = get_engine() if (cfg.persist_pg and _CAN_PG) else None
    if cfg.persist_pg and not _CAN_PG:
        log.warning("Persistencia habilitada pero Timescale no disponible.")

    async def maybe_trade() -> None:
        if last["spot"] is None or last["perp"] is None:
            return
        # compute premium: perp vs spot
        edge = (last["perp"] - last["spot"]) / last["spot"]
        if abs(edge) < cfg.threshold:
            return
        async with trade_lock:
            # recompute inside lock to avoid race
            edge = (last["perp"] - last["spot"]) / last["spot"]
            if abs(edge) < cfg.threshold:
                return

            qty = cfg.notional / last["spot"]
            ts = datetime.now(timezone.utc)

            # persist opportunity
            if engine is not None:
                try:
                    insert_cross_signal(
                        engine,
                        symbol=cfg.symbol,
                        spot_exchange=cfg.spot.name,
                        perp_exchange=cfg.perp.name,
                        spot_px=last["spot"],
                        perp_px=last["perp"],
                        edge=edge,
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar cross_signal: %s", e)

            # determine sides
            if edge > 0:
                spot_side, perp_side = "buy", "sell"
            else:
                spot_side, perp_side = "sell", "buy"

            # execute offsetting orders
            resp_spot, resp_perp = await asyncio.gather(
                cfg.spot.place_order(cfg.symbol, spot_side, "market", qty),
                cfg.perp.place_order(cfg.symbol, perp_side, "market", qty),
            )

            balances[cfg.spot.name] += qty if spot_side == "buy" else -qty
            balances[cfg.perp.name] += qty if perp_side == "buy" else -qty

            if engine is not None:
                try:
                    insert_fill(
                        engine,
                        ts=ts,
                        venue=cfg.spot.name,
                        strategy="cross_arbitrage",
                        symbol=cfg.symbol,
                        side=spot_side,
                        type_="market",
                        qty=qty,
                        price=resp_spot.get("price"),
                        fee_usdt=None,
                        raw=resp_spot,
                    )
                    insert_fill(
                        engine,
                        ts=ts,
                        venue=cfg.perp.name,
                        strategy="cross_arbitrage",
                        symbol=cfg.symbol,
                        side=perp_side,
                        type_="market",
                        qty=qty,
                        price=resp_perp.get("price"),
                        fee_usdt=None,
                        raw=resp_perp,
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar fills: %s", e)

            log.info(
                "CROSS ARB edge=%.4f%% spot=%.2f perp=%.2f qty=%.6f pos_spot=%.6f pos_perp=%.6f",
                edge * 100,
                last["spot"],
                last["perp"],
                qty,
                balances[cfg.spot.name],
                balances[cfg.perp.name],
            )

    async def listen_spot():
        async for t in cfg.spot.stream_trades(cfg.symbol):
            px = t.get("price")
            if px is not None:
                last["spot"] = float(px)
                await maybe_trade()

    async def listen_perp():
        async for t in cfg.perp.stream_trades(cfg.symbol):
            px = t.get("price")
            if px is not None:
                last["perp"] = float(px)
                await maybe_trade()

    await asyncio.gather(listen_spot(), listen_perp())
