from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from ..adapters.base import ExchangeAdapter
from ..execution.balance import rebalance_between_exchanges
from ..risk.manager import RiskManager

try:
    from ..storage.timescale import (
        get_engine,
        insert_cross_signal,
        insert_fill,
        insert_portfolio_snapshot,
    )
    _CAN_PG = True
except Exception:  # pragma: no cover - Timescale optional
    _CAN_PG = False

log = logging.getLogger(__name__)


@dataclass
class CrossArbConfig:
    """Configuration for cross exchange spot/perp arbitrage.

    Attributes
    ----------
    symbol:
        Trading pair, e.g. ``"BTC/USDT"``.
    spot:
        Spot market connector implementing :class:`ExchangeAdapter`.
    perp:
        Perpetual market connector implementing :class:`ExchangeAdapter`.
    threshold:
        Premium threshold as decimal (``0.001`` = 0.1%).
    notional:
        Quote currency notional per leg.
    persist_pg:
        If ``True`` persist signals/fills in TimescaleDB.
    rebalance_assets:
        Optional assets to periodically rebalance.
    rebalance_threshold:
        Minimum imbalance required to trigger a rebalance.
    latency:
        Simulated latency in seconds before sending orders.
    max_notional:
        Maximum quote notional allowed per trade. ``None`` disables the cap.
    max_qty:
        Maximum base quantity allowed per trade. ``None`` disables the cap.
    """

    symbol: str
    spot: ExchangeAdapter
    perp: ExchangeAdapter
    threshold: float = 0.001  # premium threshold as decimal (0.001 = 0.1%)
    notional: float = 100.0  # quote currency notional per leg
    persist_pg: bool = False
    rebalance_assets: tuple[str, ...] = ()
    rebalance_threshold: float = 0.0
    latency: float = 0.0
    max_notional: Optional[float] = None
    max_qty: Optional[float] = None


async def run_cross_exchange_arbitrage(cfg: CrossArbConfig) -> None:
    """Run a simple spot/perp cross exchange arbitrage loop.

    The function listens to trade streams from both adapters. When the
    difference between perp and spot price exceeds ``cfg.threshold`` it sends
    offsetting market orders on each venue and optionally persists the
    opportunity and resulting fills into TimescaleDB.

    The trade size can be capped via ``cfg.max_notional`` or ``cfg.max_qty``
    and a simulated network latency can be injected via ``cfg.latency``. Prior
    to submitting orders the required balances on both venues are verified.
    """

    last: Dict[str, Optional[float]] = {"spot": None, "perp": None}
    balances: Dict[str, float] = {cfg.spot.name: 0.0, cfg.perp.name: 0.0}
    trade_lock = asyncio.Lock()

    engine = get_engine() if (cfg.persist_pg and _CAN_PG) else None
    if cfg.persist_pg and not _CAN_PG:
        log.warning("Persistencia habilitada pero Timescale no disponible.")
    risk = RiskManager(equity_pct=1.0, risk_pct=0.0)

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
            if cfg.max_notional is not None:
                qty = min(qty, cfg.max_notional / last["spot"])
            if cfg.max_qty is not None:
                qty = min(qty, cfg.max_qty)
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

            base, quote = cfg.symbol.split("/")
            fetch_spot = getattr(cfg.spot, "fetch_balance", None)
            fetch_perp = getattr(cfg.perp, "fetch_balance", None)
            bal_spot, bal_perp = {}, {}
            if fetch_spot or fetch_perp:
                bal_spot, bal_perp = await asyncio.gather(
                    fetch_spot() if fetch_spot else asyncio.sleep(0),
                    fetch_perp() if fetch_perp else asyncio.sleep(0),
                )
                bal_spot = bal_spot or {}
                bal_perp = bal_perp or {}

            def _has(bal: dict, asset: str, needed: float) -> bool:
                return float(bal.get(asset, 0.0)) >= needed

            need_spot = qty * last["spot"] if spot_side == "buy" else qty
            asset_spot = quote if spot_side == "buy" else base
            need_perp = qty * last["perp"] if perp_side == "buy" else qty
            asset_perp = quote if perp_side == "buy" else base

            if not (_has(bal_spot, asset_spot, need_spot) and _has(bal_perp, asset_perp, need_perp)):
                log.debug("Trade skipped due to insufficient balance")
                return

            if cfg.latency:
                await asyncio.sleep(cfg.latency)

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
                    insert_portfolio_snapshot(
                        engine,
                        venue=cfg.spot.name,
                        symbol=cfg.symbol,
                        position=balances[cfg.spot.name],
                        price=last["spot"],
                        notional_usd=balances[cfg.spot.name] * last["spot"],
                    )
                    insert_portfolio_snapshot(
                        engine,
                        venue=cfg.perp.name,
                        symbol=cfg.symbol,
                        position=balances[cfg.perp.name],
                        price=last["perp"],
                        notional_usd=balances[cfg.perp.name] * last["perp"],
                    )
                except Exception as e:  # pragma: no cover - logging only
                    log.debug("No se pudo insertar fills/snapshot: %s", e)

            log.info(
                "CROSS ARB edge=%.4f%% spot=%.2f perp=%.2f qty=%.6f pos_spot=%.6f pos_perp=%.6f",
                edge * 100,
                last["spot"],
                last["perp"],
                qty,
                balances[cfg.spot.name],
                balances[cfg.perp.name],
            )

            if engine is not None and cfg.rebalance_assets:
                venues = {cfg.spot.name: cfg.spot, cfg.perp.name: cfg.perp}
                base, quote = cfg.symbol.split("/")
                for asset in cfg.rebalance_assets:
                    if asset == base:
                        price_ref = last["spot"]
                    elif asset == quote:
                        price_ref = 1.0
                    else:
                        price_ref = last["spot"]
                    try:
                        await rebalance_between_exchanges(
                            asset,
                            price=float(price_ref),
                            venues=venues,
                            risk=risk,
                            engine=engine,
                            threshold=cfg.rebalance_threshold,
                        )
                    except Exception as e:  # pragma: no cover - logging only
                        log.debug("No se pudo rebalancear %s: %s", asset, e)

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
