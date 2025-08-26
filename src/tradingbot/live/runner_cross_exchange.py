from __future__ import annotations

"""Simple live runner for cross exchange spot/perp arbitrage using ``ExecutionRouter``."""

import asyncio
import logging
from typing import Dict, Optional

from ..bus import EventBus
from ..execution.order_types import Order
from ..execution.router import ExecutionRouter
from ..strategies.cross_exchange_arbitrage import CrossArbConfig
from ..data.funding import poll_funding
from ..data.open_interest import poll_open_interest
from ..data.basis import poll_basis
from ..risk.manager import RiskManager, load_positions
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..risk.oco import OcoBook, load_active_oco

try:
    from ..storage.timescale import get_engine
    _CAN_PG = True
except Exception:  # pragma: no cover
    _CAN_PG = False

log = logging.getLogger(__name__)


async def run_cross_exchange(cfg: CrossArbConfig, risk: RiskService | None = None,
                            equity_pct: float = 0.0, risk_pct: float = 0.0) -> None:
    """Run cross exchange arbitrage using ``ExecutionRouter``.

    Parameters
    ----------
    cfg:
        Configuration with symbol, adapters and trading params.
    """

    router = ExecutionRouter([cfg.spot, cfg.perp])
    bus = EventBus()
    if risk is None:
        risk = RiskService(
            RiskManager(equity_pct=equity_pct, risk_pct=risk_pct),
            PortfolioGuard(GuardConfig(venue="cross")),
            daily=None,
        )

    engine = get_engine() if _CAN_PG else None
    oco_book = OcoBook()
    if engine is not None:
        venue = risk.guard.cfg.venue if getattr(risk, "guard", None) else "cross"
        pos_map = load_positions(engine, venue)
        for sym, data in pos_map.items():
            risk.update_position(venue, sym, data.get("qty", 0.0))
        oco_book.preload(load_active_oco(engine, venue=venue, symbols=[cfg.symbol]))

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
        ok1, _r1, delta1 = risk.check_order(
            cfg.symbol,
            spot_side,
            last["spot"],
            strength=qty,
        )
        ok2, _r2, delta2 = risk.check_order(
            cfg.symbol,
            perp_side,
            last["perp"],
            strength=qty,
        )
        if not (ok1 and ok2):
            return
        order_spot = Order(cfg.symbol, spot_side, "market", abs(delta1))
        order_perp = Order(cfg.symbol, perp_side, "market", abs(delta2))
        await asyncio.gather(
            router.execute(order_spot),
            router.execute(order_perp),
        )
        risk.on_fill(cfg.symbol, spot_side, qty, venue=getattr(cfg.spot, "name", "spot"))
        risk.on_fill(cfg.symbol, perp_side, qty, venue=getattr(cfg.perp, "name", "perp"))
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
            risk.mark_price(cfg.symbol, price)
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
            risk.mark_price(cfg.symbol, price)
            await maybe_trade()

    tasks = [listen_spot(), listen_perp()]
    if getattr(cfg.perp, "fetch_funding", None):
        tasks.append(poll_funding(cfg.perp, cfg.symbol, bus, persist_pg=True))
    if getattr(cfg.perp, "fetch_open_interest", None) or getattr(cfg.perp, "fetch_oi", None):
        tasks.append(poll_open_interest(cfg.perp, cfg.symbol, bus, persist_pg=True))
    if getattr(cfg.perp, "fetch_basis", None):
        tasks.append(poll_basis(cfg.perp, cfg.symbol, bus, persist_pg=True))

    await asyncio.gather(*tasks)


__all__ = ["run_cross_exchange"]

