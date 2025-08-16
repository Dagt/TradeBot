from __future__ import annotations

"""Simple live runner for cross exchange spot/perp arbitrage using ``ExecutionRouter``."""

import asyncio
import logging
from typing import Dict, Optional

from ..execution.order_types import Order
from ..execution.router import ExecutionRouter
from ..strategies.cross_exchange_arbitrage import CrossArbConfig

log = logging.getLogger(__name__)


async def run_cross_exchange(cfg: CrossArbConfig) -> None:
    """Run cross exchange arbitrage using ``ExecutionRouter``.

    Parameters
    ----------
    cfg:
        Configuration with symbol, adapters and trading params.
    """

    router = ExecutionRouter([cfg.spot, cfg.perp])

    last: Dict[str, Optional[float]] = {"spot": None, "perp": None}

    async def maybe_trade() -> None:
        if last["spot"] is None or last["perp"] is None:
            return
        edge = (last["perp"] - last["spot"]) / last["spot"]
        if abs(edge) < cfg.threshold:
            return
        qty = cfg.notional / last["spot"]
        spot_side = "buy" if edge > 0 else "sell"
        perp_side = "sell" if edge > 0 else "buy"
        order_spot = Order(cfg.symbol, spot_side, "market", qty)
        order_perp = Order(cfg.symbol, perp_side, "market", qty)
        await asyncio.gather(
            router.execute(order_spot),
            router.execute(order_perp),
        )
        log.info(
            "CROSS edge=%.4f%% spot=%.2f perp=%.2f qty=%.6f",
            edge * 100,
            last["spot"],
            last["perp"],
            qty,
        )

    async def listen_spot() -> None:
        async for t in cfg.spot.stream_trades(cfg.symbol):
            px = t.get("price")
            if px is None:
                continue
            price = float(px)
            if getattr(cfg.spot, "state", None) is None:
                cfg.spot.state = type("S", (), {"last_px": {}})  # type: ignore[attr-defined]
            cfg.spot.state.last_px[cfg.symbol] = price  # type: ignore[attr-defined]
            last["spot"] = price
            await maybe_trade()

    async def listen_perp() -> None:
        async for t in cfg.perp.stream_trades(cfg.symbol):
            px = t.get("price")
            if px is None:
                continue
            price = float(px)
            if getattr(cfg.perp, "state", None) is None:
                cfg.perp.state = type("S", (), {"last_px": {}})  # type: ignore[attr-defined]
            cfg.perp.state.last_px[cfg.symbol] = price  # type: ignore[attr-defined]
            last["perp"] = price
            await maybe_trade()

    await asyncio.gather(listen_spot(), listen_perp())


__all__ = ["run_cross_exchange"]

