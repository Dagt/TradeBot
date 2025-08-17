# src/tradingbot/cli/main.py
"""Command line interface for TradingBot.

This module exposes a Typer application with a small set of
high level commands used throughout the project.  The commands
are intentionally lightweight so importing the module does not
require heavy third party dependencies until a command is
executed.

Examples
--------
The execution router supports simple algorithms directly from the
CLI by passing ``--algo`` and its parameters.  For instance::

    tradingbot run-bot --algo twap --slices 4
    tradingbot run-bot --algo vwap --volumes 1 2 3
    tradingbot run-bot --algo pov --participation-rate 0.5

These examples illustrate how an order could be executed using
TWAP, VWAP or POV strategies.
"""

from __future__ import annotations

import asyncio
import os
import sys
import typer

from ..logging_conf import setup_logging
from tradingbot.analysis.backtest_report import generate_report


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
    from ..data.ingestion import run_orderbook_stream
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


@app.command("daemon")
def run_daemon(config: str = "config/config.yaml") -> None:
    """Launch the :class:`TradeBotDaemon` using a Hydra configuration."""

    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    setup_logging()

    # Register dataclasses and Hydra config
    from ..config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    @hydra.main(config_path=rel_path, config_name=cfg_path.stem, version_base=None)
    def _run(cfg) -> None:  # type: ignore[override]
        from ..adapters.binance_spot_ws import BinanceSpotWSAdapter
        from ..bus import EventBus
        from ..live.daemon import TradeBotDaemon
        from ..risk.manager import RiskManager
        from ..strategies.breakout_atr import BreakoutATR
        from ..execution.router import ExecutionRouter

        adapter = BinanceSpotWSAdapter()
        bus = EventBus()
        risk = RiskManager(bus=bus)
        strat = BreakoutATR()
        router = ExecutionRouter(adapters=[adapter])

        corr_thr = getattr(getattr(cfg, "risk", {}), "correlation_threshold", 0.8)
        ret_win = getattr(getattr(cfg, "risk", {}), "returns_window", 100)
        bal_assets = getattr(getattr(cfg, "balance", {}), "assets", [])
        bal_thr = getattr(getattr(cfg, "balance", {}), "threshold", 0.0)

        bot = TradeBotDaemon(
            {"binance": adapter},
            [strat],
            risk,
            router,
            [cfg.backtest.symbol],
            correlation_threshold=corr_thr,
            returns_window=ret_win,
            rebalance_assets=bal_assets,
            rebalance_threshold=bal_thr,
        )

        typer.echo(OmegaConf.to_yaml(cfg))
        asyncio.run(bot.run())

    old = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        _run()
    finally:
        sys.argv = old


@app.command("run-bybit-spot-testnet")
def run_bybit_spot_testnet(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
) -> None:
    """Run the live trading bot on Bybit spot testnet."""

    setup_logging()
    from ..live.runner_spot_testnet_bybit import run_live_bybit_spot_testnet

    asyncio.run(
        run_live_bybit_spot_testnet(
            symbol=symbol,
            trade_qty=trade_qty,
            persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt,
            per_symbol_cap_usdt=per_symbol_cap_usdt,
            soft_cap_pct=soft_cap_pct,
            soft_cap_grace_sec=soft_cap_grace_sec,
        )
    )


@app.command("run-okx-spot-testnet")
def run_okx_spot_testnet(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30,
) -> None:
    """Run the live trading bot on OKX spot testnet."""

    setup_logging()
    from ..live.runner_spot_testnet_okx import run_live_okx_spot_testnet

    asyncio.run(
        run_live_okx_spot_testnet(
            symbol=symbol,
            trade_qty=trade_qty,
            persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt,
            per_symbol_cap_usdt=per_symbol_cap_usdt,
            soft_cap_pct=soft_cap_pct,
            soft_cap_grace_sec=soft_cap_grace_sec,
        )
    )


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
    typer.echo(generate_report(result))


@app.command("backtest-cfg")
def backtest_cfg(config: str) -> None:
    """Run a backtest using a Hydra YAML configuration."""

    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    setup_logging()
    # Register dataclasses and load the YAML file
    from ..config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    @hydra.main(
        config_path=rel_path,
        config_name=cfg_path.stem,
        version_base=None,
    )
    def _run(cfg) -> None:  # type: ignore[override]
        from ..backtest.event_engine import run_backtest_csv

        data = cfg.backtest.data
        symbol = cfg.backtest.symbol
        strategy = cfg.backtest.strategy

        result = run_backtest_csv({symbol: data}, [(strategy, symbol)])
        typer.echo(OmegaConf.to_yaml(cfg))
        typer.echo(result)
        typer.echo(generate_report(result))
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        _run()
    finally:
        sys.argv = old_argv


@app.command("walk-forward")
def walk_forward_cfg(config: str) -> None:
    """Run walk-forward optimization from a Hydra configuration."""

    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    setup_logging()
    from ..config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    @hydra.main(config_path=rel_path, config_name=cfg_path.stem, version_base=None)
    def _run(cfg) -> None:  # type: ignore[override]
        from ..backtesting.engine import walk_forward_optimize

        wf_cfg = cfg.walk_forward
        results = walk_forward_optimize(
            wf_cfg.data,
            wf_cfg.symbol,
            wf_cfg.strategy,
            wf_cfg.param_grid,
            latency=getattr(wf_cfg, "latency", 1),
            window=getattr(wf_cfg, "window", 120),
            n_splits=getattr(wf_cfg, "n_splits", 3),
            embargo=getattr(wf_cfg, "embargo", 0.0),
        )
        typer.echo(OmegaConf.to_yaml(cfg))
        typer.echo(results)

    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        _run()
    finally:
        sys.argv = old_argv


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
    route: str = typer.Argument(..., help="Ruta BASE-MID-QUOTE, ej. BTC-ETH-USDT"),
    notional: float = typer.Option(100.0, help="Notional en la divisa quote"),
) -> None:
    """Ejecutar arbitrage triangular simple en Binance."""

    setup_logging()
    from ..live.runner_triangular import TriConfig, run_triangular_binance
    from ..strategies.arbitrage_triangular import TriRoute

    try:
        base, mid, quote = route.split("-")
    except ValueError as exc:  # pragma: no cover - validated por typer
        raise typer.BadParameter("Formato de ruta inválido, usa BASE-MID-QUOTE") from exc

    cfg = TriConfig(route=TriRoute(base, mid, quote), notional_quote=notional)
    asyncio.run(run_triangular_binance(cfg))


@app.command("cross-arb")
def cross_arb(
    symbol: str = typer.Argument("BTC/USDT", help="Símbolo a arbitrar"),
    spot: str = typer.Argument(..., help="Adapter spot, ej. binance_spot"),
    perp: str = typer.Argument(..., help="Adapter perp, ej. binance_futures"),
    threshold: float = typer.Option(0.001, help="Umbral de premium (decimales)"),
    notional: float = typer.Option(100.0, help="Notional por pata en moneda quote"),
) -> None:
    """Arbitraje entre spot y perp usando dos adapters."""

    setup_logging()
    from ..adapters import (
        BinanceFuturesAdapter,
        BinanceSpotAdapter,
        BybitFuturesAdapter,
        BybitSpotAdapter,
        OKXFuturesAdapter,
        OKXSpotAdapter,
    )
    from ..strategies.cross_exchange_arbitrage import (
        CrossArbConfig,
        run_cross_exchange_arbitrage,
    )

    adapters = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "bybit_spot": BybitSpotAdapter,
        "bybit_futures": BybitFuturesAdapter,
        "okx_spot": OKXSpotAdapter,
        "okx_futures": OKXFuturesAdapter,
    }

    if spot not in adapters or perp not in adapters:
        choices = ", ".join(sorted(adapters))
        raise typer.BadParameter(f"Adapters válidos: {choices}")

    cfg = CrossArbConfig(
        symbol=symbol,
        spot=adapters[spot](),
        perp=adapters[perp](),
        threshold=threshold,
        notional=notional,
    )
    asyncio.run(run_cross_exchange_arbitrage(cfg))


@app.command("run-cross-arb")
def run_cross_arb(
    symbol: str = typer.Argument("BTC/USDT", help="Símbolo a arbitrar"),
    spot: str = typer.Argument(..., help="Adapter spot, ej. binance_spot"),
    perp: str = typer.Argument(..., help="Adapter perp, ej. binance_futures"),
    threshold: float = typer.Option(0.001, help="Umbral de premium (decimales)"),
    notional: float = typer.Option(100.0, help="Notional por pata en moneda quote"),
) -> None:
    """Ejecuta el runner de arbitraje spot/perp con ``ExecutionRouter``."""

    setup_logging()
    from ..adapters import (
        BinanceFuturesAdapter,
        BinanceSpotAdapter,
        BybitFuturesAdapter,
        BybitSpotAdapter,
        OKXFuturesAdapter,
        OKXSpotAdapter,
    )
    from ..strategies.cross_exchange_arbitrage import CrossArbConfig
    from ..live.runner_cross_exchange import run_cross_exchange

    adapters = {
        "binance_spot": BinanceSpotAdapter,
        "binance_futures": BinanceFuturesAdapter,
        "bybit_spot": BybitSpotAdapter,
        "bybit_futures": BybitFuturesAdapter,
        "okx_spot": OKXSpotAdapter,
        "okx_futures": OKXFuturesAdapter,
    }

    if spot not in adapters or perp not in adapters:
        choices = ", ".join(sorted(adapters))
        raise typer.BadParameter(f"Adapters válidos: {choices}")

    cfg = CrossArbConfig(
        symbol=symbol,
        spot=adapters[spot](),
        perp=adapters[perp](),
        threshold=threshold,
        notional=notional,
    )
    asyncio.run(run_cross_exchange(cfg))


def main() -> None:
    """Entry point used by ``python -m tradingbot.cli``."""

    app()


if __name__ == "__main__":  # pragma: no cover
    main()

