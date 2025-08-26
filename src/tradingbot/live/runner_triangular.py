from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict

import websockets

from ..utils.metrics import WS_FAILURES

from ..execution.paper import PaperAdapter
from ..strategies.arbitrage_triangular import (
    TriRoute, make_symbols, compute_edge
)
from ..risk.manager import RiskManager, load_positions
from ..risk.portfolio_guard import PortfolioGuard, GuardConfig
from ..risk.service import RiskService
from ..risk.oco import OcoBook, load_active_oco

# Persistencia opcional
try:
    from ..storage.timescale import get_engine, insert_order, insert_tri_signal
    _CAN_PG = True
except Exception:
    _CAN_PG = False

log = logging.getLogger(__name__)

BINANCE_WS = "wss://stream.binance.com:9443/stream?streams="

def _stream_name(sym: str) -> str:
    return sym.replace("/", "").lower() + "@trade"

@dataclass
class TriConfig:
    route: TriRoute
    taker_fee_bps: float = 7.5
    buffer_bps: float = 3.0
    edge_threshold: float = 0.001
    persist_pg: bool = False  # <-- nuevo flag

async def run_triangular_binance(cfg: TriConfig, risk: RiskService | None = None):
    syms = make_symbols(cfg.route)
    streams = [_stream_name(syms.bq), _stream_name(syms.mq), _stream_name(syms.mb)]
    url = BINANCE_WS + "/".join(streams)

    broker = PaperAdapter(fee_bps=cfg.taker_fee_bps)
    last: Dict[str, float] = {"bq": None, "mq": None, "mb": None}
    fills = 0
    if risk is None:
        risk = RiskService(
            RiskManager(risk_pct=0.0),
            PortfolioGuard(GuardConfig(venue="binance")),
        )

    engine = get_engine() if (cfg.persist_pg and _CAN_PG) else None
    oco_book = OcoBook()
    if engine is not None:
        pos_map = load_positions(engine, risk.guard.cfg.venue)
        for sym, data in pos_map.items():
            risk.update_position(risk.guard.cfg.venue, sym, data.get("qty", 0.0))
        oco_book.preload(
            load_active_oco(
                engine,
                venue=risk.guard.cfg.venue,
                symbols=[syms.bq, syms.mq, syms.mb],
            )
        )
    if cfg.persist_pg and not _CAN_PG:
        log.warning("Persistencia habilitada pero SQL no disponible (sqlalchemy/psycopg2).")

    backoff = 1.0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                log.info("WS conectado (triangular): %s", url)
                backoff = 1.0
                async for raw in ws:
                    msg = json.loads(raw)
                    data = msg.get("data") or {}
                    s = msg.get("stream") or ""
                    price = data.get("p")
                    if price is None:
                        continue
                    price = float(price)

                    if syms.bq.replace("/", "").lower() in s:
                        last["bq"] = price
                        risk.mark_price(syms.bq, price)
                    elif syms.mq.replace("/", "").lower() in s:
                        last["mq"] = price
                        risk.mark_price(syms.mq, price)
                    elif syms.mb.replace("/", "").lower() in s:
                        last["mb"] = price
                        risk.mark_price(syms.mb, price)

                    if None in last.values():
                        continue

                    edge = compute_edge(last, cfg.taker_fee_bps, cfg.buffer_bps)
                    if edge is None or edge.net < cfg.edge_threshold:
                        continue

                    # Persistir señal
                    strength = min(edge.net, 1.0)
                    if strength <= 0:
                        continue

                    # Ejecutar 3 patas en PAPER
                    eq = broker.equity(
                        mark_prices={
                            f"{cfg.route.base}/{cfg.route.quote}": last["bq"],
                            f"{cfg.route.mid}/{cfg.route.base}": last["mb"],
                            f"{cfg.route.mid}/{cfg.route.quote}": last["mq"],
                        }
                    )
                    if engine is not None:
                        try:
                            insert_tri_signal(
                                engine,
                                exchange="binance",
                                base=cfg.route.base,
                                mid=cfg.route.mid,
                                quote=cfg.route.quote,
                                direction=edge.direction,
                                edge=edge.net,
                                notional_quote=eq * strength,
                                taker_fee_bps=cfg.taker_fee_bps,
                                buffer_bps=cfg.buffer_bps,
                                bq=last["bq"], mq=last["mq"], mb=last["mb"]
                            )
                        except Exception as e:
                            log.debug("No se pudo insertar tri_signal: %s", e)
                    if edge.direction == "b->m":
                        s1 = strength
                        ok1, _, d1 = risk.check_order(
                            f"{cfg.route.base}/{cfg.route.quote}",
                            "buy",
                            eq,
                            last["bq"],
                            strength=s1,
                        )
                        s2 = strength / last["bq"]
                        ok2, _, d2 = risk.check_order(
                            f"{cfg.route.mid}/{cfg.route.base}",
                            "buy",
                            eq,
                            last["mb"],
                            strength=s2,
                        )
                        s3 = strength * last["mq"] / (last["bq"] * last["mb"])
                        ok3, _, d3 = risk.check_order(
                            f"{cfg.route.mid}/{cfg.route.quote}",
                            "sell",
                            eq,
                            last["mq"],
                            strength=s3,
                        )
                        if not (ok1 and ok2 and ok3):
                            continue
                        base_qty = abs(d1)
                        mid_qty = abs(d2)
                        resp1 = await broker.place_order(
                            f"{cfg.route.base}/{cfg.route.quote}",
                            "buy",
                            "market",
                            base_qty,
                        )
                        resp2 = await broker.place_order(
                            f"{cfg.route.mid}/{cfg.route.base}",
                            "buy",
                            "market",
                            mid_qty,
                        )
                        resp3 = await broker.place_order(
                            f"{cfg.route.mid}/{cfg.route.quote}",
                            "sell",
                            "market",
                            abs(d3),
                        )
                        legs = [
                            ("buy", f"{cfg.route.base}/{cfg.route.quote}", base_qty, resp1),
                            ("buy", f"{cfg.route.mid}/{cfg.route.base}", mid_qty, resp2),
                            ("sell", f"{cfg.route.mid}/{cfg.route.quote}", abs(d3), resp3),
                        ]
                    else:
                        s1 = strength
                        ok1, _, d1 = risk.check_order(
                            f"{cfg.route.mid}/{cfg.route.quote}",
                            "buy",
                            eq,
                            last["mq"],
                            strength=s1,
                        )
                        s2 = strength * last["mb"] / last["mq"]
                        ok2, _, d2 = risk.check_order(
                            f"{cfg.route.mid}/{cfg.route.base}",
                            "sell",
                            eq,
                            last["mb"],
                            strength=s2,
                        )
                        s3 = strength * last["bq"] / (last["mq"] * last["mb"])
                        ok3, _, d3 = risk.check_order(
                            f"{cfg.route.base}/{cfg.route.quote}",
                            "sell",
                            eq,
                            last["bq"],
                            strength=s3,
                        )
                        if not (ok1 and ok2 and ok3):
                            continue
                        mid_qty = abs(d1)
                        base_qty = abs(d3)
                        resp1 = await broker.place_order(
                            f"{cfg.route.mid}/{cfg.route.quote}",
                            "buy",
                            "market",
                            mid_qty,
                        )
                        resp2 = await broker.place_order(
                            f"{cfg.route.mid}/{cfg.route.base}",
                            "sell",
                            "market",
                            abs(d2),
                        )
                        resp3 = await broker.place_order(
                            f"{cfg.route.base}/{cfg.route.quote}",
                            "sell",
                            "market",
                            base_qty,
                        )
                        legs = [
                            ("buy", f"{cfg.route.mid}/{cfg.route.quote}", mid_qty, resp1),
                            ("sell", f"{cfg.route.mid}/{cfg.route.base}", abs(d2), resp2),
                            ("sell", f"{cfg.route.base}/{cfg.route.quote}", base_qty, resp3),
                        ]

                    fills += 3
                    for side, sym, qty_leg, _r in legs:
                        risk.on_fill(sym, side, qty_leg, venue="binance")
                    # Persistir órdenes paper
                    if engine is not None:
                        for side, symbol, qty, r in legs:
                            try:
                                insert_order(
                                    engine,
                                    strategy="tri_arbitrage",
                                    exchange="binance",
                                    symbol=symbol,
                                    side=side,
                                    type_="market",
                                    qty=float(qty),
                                    px=float(r.get("price")) if r.get("price") is not None else None,
                                    status=r.get("status","filled"),
                                    ext_order_id=r.get("order_id"),
                                    notes=r
                                )
                            except Exception as e:
                                log.debug("No se pudo insertar order: %s", e)

                    eq = broker.equity(mark_prices={
                        f"{cfg.route.base}/{cfg.route.quote}": last["bq"],
                        f"{cfg.route.mid}/{cfg.route.quote}": last["mq"],
                        f"{cfg.route.mid}/{cfg.route.base}": last["mb"],
                    })
                    log.info(
                        "ARBITRAJE %s edge=%.4f%% strength=%.4f %s | fills=%d | equity=%.4f",
                        edge.direction, edge.net*100, strength, last, fills, eq
                    )
        except Exception as e:
            WS_FAILURES.labels(adapter="tri_runner").inc()
            log.warning("WS triangular desconectado (%s). Reintentando en %.1fs ...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
