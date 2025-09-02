"""Market data related CLI commands."""
from __future__ import annotations

import asyncio
from typing import List

import click
import typer

from ...logging_conf import setup_logging
from .. import utils

app = typer.Typer(help="Data utilities")

# Exchanges supported for backfill command
BACKFILL_EXCHANGES = [
    "binance_spot",
    "binance_futures",
    "okx_spot",
    "okx_futures",
    "bybit_spot",
    "bybit_futures",
    "deribit_futures",
]


@app.command()
def ingest(
    venue: str = typer.Option(
        "binance_spot", help=f"Data venue adapter ({', '.join(sorted(utils._AVAILABLE_VENUES))})"
    ),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Market symbols"),
    depth: int = typer.Option(10, help="Order book depth"),
    kind: str = typer.Option(
        "orderbook",
        "--kind",
        help="Data kind: trades,trades_multi,orderbook,bba,delta,funding,oi,bars",
    ),
    persist: bool = typer.Option(False, "--persist", help="Persist data"),
    backend: str = typer.Option("timescale", "--backend"),
    timeframe: str = typer.Option(
        "3m", "--timeframe", help="Bar timeframe like 1m, 3m, 5m (kind=bars only)"
    ),
) -> None:
    """Stream market data from a venue and optionally persist it."""

    setup_logging()

    from ...bus import EventBus
    from ...data import ingestion as ing
    from ...types import Tick
    from ...storage import quest as qs_storage, timescale as ts_storage

    if venue not in utils._AVAILABLE_VENUES:
        choices = ", ".join(sorted(utils._AVAILABLE_VENUES))
        raise typer.BadParameter(f"Invalid venue, choose one of: {choices}")

    adapter_class = utils.get_adapter_class(venue)
    if adapter_class is None:
        choices = ", ".join(sorted(utils._AVAILABLE_VENUES))
        raise typer.BadParameter(f"Invalid venue, choose one of: {choices}")
    adapter = adapter_class()

    alias_kind = "open_interest" if kind == "oi" else kind
    supported_kinds = utils.get_supported_kinds(adapter_class)
    if alias_kind not in supported_kinds:
        choices = ", ".join(sorted(supported_kinds))
        raise typer.BadParameter(
            f"adapter does not support {alias_kind} (supported: {choices})"
        )
    kind = alias_kind

    from ...core import normalize

    symbols = [normalize(s) for s in symbols]
    bus = EventBus()
    engine = None
    if persist and backend != "csv":
        storage = ts_storage if backend == "timescale" else qs_storage
        engine = storage.get_engine()

    if kind == "orderbook":
        bus.subscribe("orderbook", lambda ob: typer.echo(str(ob)))

    async def _run() -> None:
        if hasattr(adapter, "_configure_mode"):
            await adapter._configure_mode()
        tasks = []
        for sym in symbols:
            if kind == "orderbook":
                if persist and backend == "csv":
                    tasks.append(
                        asyncio.create_task(
                            ing.stream_orderbook(
                                adapter, sym, depth, backend="csv"
                            )
                        )
                    )
                else:
                    tasks.append(
                        asyncio.create_task(
                            ing.run_orderbook_stream(
                                adapter,
                                sym,
                                depth,
                                bus,
                                engine,
                                persist=persist,
                                backend=backend,
                            )
                        )
                    )
            elif kind == "trades":
                async def _t(symbol: str) -> None:
                    async for d in adapter.stream_trades(symbol):
                        typer.echo(str(d))
                        if persist:
                            from ...core import normalize

                            db_symbol = normalize(symbol)
                            tick = Tick(
                                ts=d.get("ts"),
                                exchange=adapter.name,
                                symbol=db_symbol,
                                price=float(d.get("price") or 0.0),
                                qty=float(d.get("qty") or 0.0),
                                side=d.get("side"),
                            )
                            ing.persist_trades([tick], backend=backend)

                tasks.append(asyncio.create_task(_t(sym)))
            elif kind == "trades_multi":
                async def _tm() -> None:
                    async for d in adapter.stream_trades_multi(symbols):
                        typer.echo(str(d))
                        if persist:
                            from ...core import normalize

                            db_symbol = normalize(d.get("symbol", ""))
                            tick = Tick(
                                ts=d.get("ts"),
                                exchange=adapter.name,
                                symbol=db_symbol,
                                price=float(d.get("price") or 0.0),
                                qty=float(d.get("qty") or 0.0),
                                side=d.get("side"),
                            )
                            ing.persist_trades([tick], backend=backend)

                tasks.append(asyncio.create_task(_tm()))
                break
            elif kind == "bba":
                async def _b(symbol: str) -> None:
                    async for d in adapter.stream_bba(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": normalize(symbol)})
                            ing.persist_bba([data], backend=backend)

                tasks.append(asyncio.create_task(_b(sym)))
            elif kind == "delta":
                async def _d(symbol: str) -> None:
                    async for d in adapter.stream_book_delta(symbol, depth):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": normalize(symbol)})
                            ing.persist_book_delta([data], backend=backend)

                tasks.append(asyncio.create_task(_d(sym)))
            elif kind == "funding":
                async def _f(symbol: str) -> None:
                    async for d in adapter.stream_funding(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_funding([data], backend=backend)

                tasks.append(asyncio.create_task(_f(sym)))
            elif kind == "open_interest":
                async def _oi(symbol: str) -> None:
                    async for d in adapter.stream_open_interest(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_open_interest([data], backend=backend)

                tasks.append(asyncio.create_task(_oi(sym)))
            elif kind == "bars":
                tasks.append(
                    asyncio.create_task(
                        ing.fetch_bars(
                            adapter,
                            sym,
                            timeframe=timeframe,
                            backend=backend,
                            persist=persist,
                        )
                    )
                )
            else:  # pragma: no cover - CLI validation
                raise typer.BadParameter("invalid kind")

        if tasks:
            await asyncio.gather(*tasks)

    typer.echo(
        f"Streaming {', '.join(symbols)} {kind} from {venue} ... (Ctrl+C to stop)"
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo("stopped")
    finally:
        if engine is not None:
            engine.dispose()


@app.command()
def backfill(
    days: int = typer.Option(1, "--days", help="Number of days to backfill"),
    symbols: List[str] = typer.Option(
        ["BTC/USDT"],
        "--symbols",
        help="Symbols to download. Accepts values without separators and normalizes them for CCXT.",
    ),
    exchange: str = typer.Option(
        "binance_spot",
        "--exchange",
        "--exchange-name",
        "--venue",
        type=click.Choice(BACKFILL_EXCHANGES),
        help=(
            "Exchange to backfill. Choose from: "
            f"{', '.join(BACKFILL_EXCHANGES)}"
        ),
    ),
    start: str | None = typer.Option(
        None, "--start", help="Start datetime in ISO format"
    ),
    end: str | None = typer.Option(
        None, "--end", help="End datetime in ISO format"
    ),
    timeframe: str = typer.Option(
        "3m", "--timeframe", help="1m, 2m, 3m, 5m, 15m, 30m, 1h, 4h, ..."
    ),
) -> None:
    """Backfill OHLCV and trades for symbols with rate limiting."""

    setup_logging()
    from datetime import datetime, timezone
    from ...jobs.backfill import backfill as run_backfill

    def _parse(dt: str | None) -> datetime | None:
        if dt is None:
            return None
        parsed = datetime.fromisoformat(dt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    timeframe = timeframe.lower()
    asyncio.run(
        run_backfill(
            days=days,
            symbols=symbols,
            exchange_name=exchange,
            start=_parse(start),
            end=_parse(end),
            timeframe=timeframe,
        )
    )


@app.command("ingest-historical")
def ingest_historical(
    source: str = typer.Argument(..., help="Fuente de datos: kaiko o coinapi"),
    symbol: str = typer.Argument(..., help="Símbolo o par"),
    exchange: str = typer.Option("", help="Exchange para Kaiko"),
    kind: str = typer.Option(
        "trades",
        help="Tipo de dato: trades, orderbook, bba, book_delta, open_interest o funding",
    ),
    backend: str = typer.Option("timescale", help="Backend de storage"),
    limit: int = typer.Option(100, help="Límite de trades"),
    depth: int = typer.Option(10, help="Profundidad del order book"),
    start: str | None = typer.Option(None, "--start", help="Inicio ISO8601"),
    end: str | None = typer.Option(None, "--end", help="Fin ISO8601"),
) -> None:
    """Ingest historical data from external providers."""

    setup_logging()

    from datetime import datetime
    from ...connectors.kaiko import KaikoConnector
    from ...connectors.coinapi import CoinAPIConnector
    from ...data import ingestion as ing
    from ...core import normalize

    if source == "kaiko":
        try:
            conn = KaikoConnector()
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        if kind == "trades":
            data = conn.fetch_trades(symbol, limit=limit)
            ticks = [
                {
                    "ts": datetime.fromtimestamp(d[0] / 1e3),
                    "price": d[1],
                    "qty": d[2],
                    "side": d[3],
                }
                for d in data
            ]
            ing.persist_trades(
                [
                    ing.Tick(
                        ts=t["ts"],
                        exchange=exchange,
                        symbol=normalize(symbol),
                        price=t["price"],
                        qty=t["qty"],
                        side=t["side"],
                    )
                    for t in ticks
                ],
                backend=backend,
            )
        else:  # pragma: no cover - CLI validation
            raise typer.BadParameter("Tipo de dato no soportado para Kaiko")
    elif source == "coinapi":
        try:
            conn = CoinAPIConnector()
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1)
        if kind == "trades":
            data = conn.fetch_trades(symbol, limit=limit, start=start, end=end)
            ticks = [
                {
                    "ts": datetime.fromisoformat(d["time_exchange"]),
                    "price": d["price"],
                    "qty": d["size"],
                    "side": d.get("taker_side"),
                }
                for d in data
            ]
            ing.persist_trades(
                [
                    ing.Tick(
                        ts=t["ts"],
                        exchange="coinapi",
                        symbol=normalize(symbol),
                        price=t["price"],
                        qty=t["qty"],
                        side=t["side"],
                    )
                    for t in ticks
                ],
                backend=backend,
            )
        elif kind == "orderbook":
            data = conn.fetch_orderbook(symbol, depth=depth, start=start, end=end)
            ing.persist_orderbook_snapshots(data, backend=backend)
        elif kind == "bba":
            data = conn.fetch_bba(symbol, start=start, end=end)
            ing.persist_bba(data, backend=backend)
        elif kind == "book_delta":
            data = conn.fetch_book_delta(symbol, depth=depth, start=start, end=end)
            ing.persist_book_delta(data, backend=backend)
        elif kind == "open_interest":
            data = conn.fetch_open_interest(symbol, start=start, end=end)
            ing.persist_open_interest(data, backend=backend)
        elif kind == "funding":
            data = conn.fetch_funding_rates(symbol, start=start, end=end)
            ing.persist_funding(data, backend=backend)
        else:  # pragma: no cover - CLI validation
            raise typer.BadParameter("Tipo de dato no soportado para CoinAPI")
    else:  # pragma: no cover - CLI validation
        raise typer.BadParameter("Fuente inválida, usa kaiko o coinapi")


@app.command("ingestion-workers")
def run_ingestion_workers(
    config: str = "config/config.yaml",
    backend: str = typer.Option("timescale", help="Storage backend"),
) -> None:
    """Start ingestion workers defined in a YAML config."""

    import yaml

    from ...workers import funding_worker, open_interest_worker

    setup_logging()

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    ingestion_cfg = cfg.get("ingestion", {})
    funding_cfg = ingestion_cfg.get("funding", {})
    oi_cfg = ingestion_cfg.get("open_interest", {})

    adapters_cache: dict[str, object] = {}

    def _load_adapter(name: str):
        if name not in adapters_cache:
            cls = utils.get_adapter_class(name)
            if cls is None:
                raise typer.BadParameter(f"Adapter {name} not found")
            adapters_cache[name] = cls()
        return adapters_cache[name]

    async def _run() -> None:
        tasks = []
        for exch, symbols in funding_cfg.items():
            adapter = _load_adapter(exch)
            if hasattr(adapter, "_configure_mode"):
                await adapter._configure_mode()
            for sym, interval in symbols.items():
                tasks.append(
                    asyncio.create_task(
                        funding_worker(adapter, sym, interval=interval, backend=backend)
                    )
                )
        for exch, symbols in oi_cfg.items():
            adapter = _load_adapter(exch)
            if hasattr(adapter, "_configure_mode"):
                await adapter._configure_mode()
            for sym, interval in symbols.items():
                tasks.append(
                    asyncio.create_task(
                        open_interest_worker(adapter, sym, interval=interval, backend=backend)
                    )
                )
        if tasks:
            await asyncio.gather(*tasks)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        typer.echo("stopped")


@app.command()
def report(venue: str = "binance_spot_testnet") -> None:
    """Display a simple PnL summary from TimescaleDB."""

    setup_logging()
    engine = None
    try:
        from ...storage.timescale import get_engine, select_pnl_summary

        engine = get_engine()
        summary = select_pnl_summary(engine, venue=venue)
    except Exception as exc:  # pragma: no cover - best effort
        summary = {"warning": str(exc)}
    finally:
        if engine is not None:
            engine.dispose()

    typer.echo(summary)

