from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict

import websockets

from ..execution.paper import PaperAdapter
from ..strategies.arbitrage_triangular import (
    TriRoute, make_symbols, compute_edge, compute_qtys_for_route
)

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
    notional_quote: float = 100.0
    edge_threshold: float = 0.001
    persist_pg: bool = False  # <-- nuevo flag

async def run_triangular_binance(cfg: TriConfig):
    syms = make_symbols(cfg.route)
    streams = [_stream_name(syms.bq), _stream_name(syms.mq), _stream_name(syms.mb)]
    url = BINANCE_WS + "/".join(streams)

    broker = PaperAdapter(fee_bps=cfg.taker_fee_bps)
    last: Dict[str, float] = {"bq": None, "mq": None, "mb": None}
    fills = 0

    engine = get_engine() if (cfg.persist_pg and _CAN_PG) else None
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
                    elif syms.mq.replace("/", "").lower() in s:
                        last["mq"] = price
                    elif syms.mb.replace("/", "").lower() in s:
                        last["mb"] = price

                    if None in last.values():
                        continue

                    edge = compute_edge(last, cfg.taker_fee_bps, cfg.buffer_bps)
                    if edge is None or edge.net < cfg.edge_threshold:
                        continue

                    # Persistir señal
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
                                notional_quote=cfg.notional_quote,
                                taker_fee_bps=cfg.taker_fee_bps,
                                buffer_bps=cfg.buffer_bps,
                                bq=last["bq"], mq=last["mq"], mb=last["mb"]
                            )
                        except Exception as e:
                            log.debug("No se pudo insertar tri_signal: %s", e)

                    # Calcular cantidades aproximadas
                    q = compute_qtys_for_route(
                        edge.direction, cfg.notional_quote, last,
                        cfg.taker_fee_bps, cfg.buffer_bps
                    )

                    # Ejecutar 3 patas en PAPER
                    if edge.direction == "b->m":
                        resp1 = await broker.place_order(f"{cfg.route.base}/{cfg.route.quote}", "buy", "market", q["base_qty"])
                        resp2 = await broker.place_order(f"{cfg.route.mid}/{cfg.route.base}", "buy", "market", q["mid_qty"])
                        resp3 = await broker.place_order(f"{cfg.route.mid}/{cfg.route.quote}", "sell", "market", q["mid_qty"])
                        legs = [
                            ("buy", f"{cfg.route.base}/{cfg.route.quote}", q["base_qty"], resp1),
                            ("buy", f"{cfg.route.mid}/{cfg.route.base}", q["mid_qty"], resp2),
                            ("sell", f"{cfg.route.mid}/{cfg.route.quote}", q["mid_qty"], resp3),
                        ]
                    else:
                        resp1 = await broker.place_order(f"{cfg.route.mid}/{cfg.route.quote}", "buy", "market", q["mid_qty"])
                        resp2 = await broker.place_order(f"{cfg.route.mid}/{cfg.route.base}", "sell", "market", q["mid_qty"])
                        resp3 = await broker.place_order(f"{cfg.route.base}/{cfg.route.quote}", "sell", "market", q["base_qty"])
                        legs = [
                            ("buy", f"{cfg.route.mid}/{cfg.route.quote}", q["mid_qty"], resp1),
                            ("sell", f"{cfg.route.mid}/{cfg.route.base}", q["mid_qty"], resp2),
                            ("sell", f"{cfg.route.base}/{cfg.route.quote}", q["base_qty"], resp3),
                        ]

                    fills += 3
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
                        "ARBITRAJE %s edge=%.4f%% notional=%.2f %s | fills=%d | equity=%.4f",
                        edge.direction, edge.net*100, cfg.notional_quote, last, fills, eq
                    )
        except Exception as e:
            log.warning("WS triangular desconectado (%s). Reintentando en %.1fs ...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
