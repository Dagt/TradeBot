# src/tradingbot/cli/main.py
"""Command line interface for TradingBot.

This module exposes a Typer application with a small set of
high level commands used throughout the project.  The commands
are intentionally lightweight so importing the module does not
require heavy third party dependencies until a command is
executed.
"""

from __future__ import annotations

import asyncio
import typer

from ..logging_conf import setup_logging


app = typer.Typer(add_completion=False, help="Utilities for running TradingBot")


@app.command()
def ingest(symbol: str = "BTC/USDT", depth: int = 10) -> None:
    """Stream order book data from Binance testnet into storage.

    The command runs until interrupted and persists snapshots using
    the Timescale helper if available.
    """

    setup_logging()
    from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
    from ..bus import EventBus
    from ..data.ingest import run_orderbook_stream
    from ..storage.timescale import get_engine

    adapter = BinanceSpotWSAdapter()
    bus = EventBus()
    engine = get_engine()

    typer.echo(f"Streaming {symbol} order book ... (Ctrl+C to stop)")
    try:
        asyncio.run(run_orderbook_stream(adapter, symbol, depth, bus, engine))
    except KeyboardInterrupt:
        typer.echo("stopped")


@app.command("run-bot")
def run_bot(symbol: str = "BTC/USDT") -> None:
    """Run the live trading bot on Binance testnet."""

    setup_logging()
    from ..live.runner import run_live_binance

    asyncio.run(run_live_binance(symbol=symbol))


@app.command()
def backtest(
    data: str,
    symbol: str = "BTC/USDT",
    strategy: str = typer.Option("breakout_atr", help="Strategy name"),
) -> None:
    """Run a simple vectorised backtest from a CSV file."""

    setup_logging()
    from ..backtest.event_engine import run_backtest_csv

    result = run_backtest_csv({symbol: data}, [(strategy, symbol)])
    typer.echo(result)


@app.command()
def report(venue: str = "binance_spot_testnet") -> None:
    """Display a simple PnL summary from TimescaleDB.

    If Timescale or its dependencies are not available the command
    will return a short warning instead of failing.
    """

    setup_logging()
    try:
        from ..storage.timescale import get_engine, select_pnl_summary

        engine = get_engine()
        summary = select_pnl_summary(engine, venue=venue)
    except Exception as exc:  # pragma: no cover - best effort
        summary = {"warning": str(exc)}

    typer.echo(summary)


@app.command("tri-arb")
def tri_arb(
    route: str = typer.Option(..., help="Ruta BASE-MID-QUOTE, p.ej. BTC-ETH-USDT"),
    notional: float = typer.Option(100.0, help="Notional en QUOTE por ciclo"),
) -> None:
    """Ejecutar arbitraje triangular simple sobre Binance."""

    setup_logging()
    from ..live.runner_triangular import TriConfig, TriRoute, run_triangular_binance

    try:
        base, mid, quote = route.split("-")
    except ValueError as exc:  # pragma: no cover - validation
        raise typer.BadParameter("route debe tener formato BASE-MID-QUOTE") from exc

    cfg = TriConfig(route=TriRoute(base=base, mid=mid, quote=quote), notional_quote=notional)
    asyncio.run(run_triangular_binance(cfg))


@app.command("cross-arb")
def cross_arb(
    symbol: str = "BTC/USDT",
    spot: str = typer.Option(..., help="Adapter spot, e.g. binance_spot"),
    perp: str = typer.Option(..., help="Adapter perp, e.g. bybit_futures"),
    threshold: float = typer.Option(0.001, help="Umbral premium como decimal"),
    notional: float = typer.Option(100.0, help="Notional en QUOTE por pata"),
) -> None:
    """Arbitraje simple spot/perp entre dos exchanges."""

    setup_logging()
    from ..strategies.cross_exchange_arbitrage import (
        CrossArbConfig,
        run_cross_exchange_arbitrage,
    )
    from ..adapters import (
        BinanceSpotAdapter,
        BinanceFuturesAdapter,
        BybitSpotAdapter,
        BybitFuturesAdapter,
        OKXSpotAdapter,
        OKXFuturesAdapter,
    )

    adapters = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "bybit_spot": BybitSpotAdapter,
        "bybit_futures": BybitFuturesAdapter,
        "okx_spot": OKXSpotAdapter,
        "okx_futures": OKXFuturesAdapter,
    }

    try:
        spot_adapter = adapters[spot.lower()]()
        perp_adapter = adapters[perp.lower()]()
    except KeyError as e:  # pragma: no cover - validation
        raise typer.BadParameter(f"adapter desconocido: {e.args[0]}")

    cfg = CrossArbConfig(
        symbol=symbol,
        spot=spot_adapter,
        perp=perp_adapter,
        threshold=threshold,
        notional=notional,
    )
    asyncio.run(run_cross_exchange_arbitrage(cfg))


def main() -> None:
    """Entry point used by ``python -m tradingbot.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

