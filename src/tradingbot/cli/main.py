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


def main() -> None:
    """Entry point used by ``python -m tradingbot.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

