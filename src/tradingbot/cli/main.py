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
import logging
import os
import sys
from typing import List

import typer

from ..logging_conf import setup_logging
from tradingbot.analysis.backtest_report import generate_report
from tradingbot.utils.time_sync import check_ntp_offset


_OFFSET_THRESHOLD = float(os.getenv("NTP_OFFSET_THRESHOLD", "1.0"))
try:
    _offset = check_ntp_offset()
    if abs(_offset) > _OFFSET_THRESHOLD:
        logging.warning(
            "System clock offset %.3fs exceeds threshold %.2fs."
            " Please synchronize your clock.",
            _offset,
            _OFFSET_THRESHOLD,
        )
except Exception as exc:  # pragma: no cover - network failures
    logging.debug("Failed to check NTP offset: %s", exc)


app = typer.Typer(add_completion=False, help="Utilities for running TradingBot")


@app.command()
def ingest(
    venue: str = typer.Option("binance_spot_ws", help="Data venue adapter"),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Market symbols"),
    depth: int = typer.Option(10, help="Order book depth"),
    kind: str = typer.Option(
        "orderbook", "--kind", help="Data kind: trades,orderbook,bba,delta,funding,oi"
    ),
    persist: bool = typer.Option(False, "--persist", help="Persist data"),
) -> None:
    """Stream market data from a venue and optionally persist it."""

    setup_logging()
    from importlib import import_module

    from ..bus import EventBus
    from ..data import ingestion as ing
    from ..types import Tick
    from ..storage.timescale import get_engine

    module = import_module(f"tradingbot.adapters.{venue}")
    cls_name = "".join(part.capitalize() for part in venue.split("_")) + "Adapter"
    adapter = getattr(module, cls_name)()

    bus = EventBus()
    engine = get_engine() if persist else None

    if kind == "orderbook":
        bus.subscribe("orderbook", lambda ob: typer.echo(str(ob)))

    async def _run() -> None:
        tasks = []
        for sym in symbols:
            if kind == "orderbook":
                tasks.append(
                    asyncio.create_task(
                        ing.run_orderbook_stream(
                            adapter,
                            sym,
                            depth,
                            bus,
                            engine,
                            persist=persist,
                        )
                    )
                )
            elif kind == "trades":
                async def _t(symbol: str) -> None:
                    async for d in adapter.stream_trades(symbol):
                        typer.echo(str(d))
                        if persist:
                            tick = Tick(
                                ts=d.get("ts"),
                                exchange=adapter.name,
                                symbol=symbol,
                                price=float(d.get("price") or 0.0),
                                qty=float(d.get("qty") or 0.0),
                                side=d.get("side"),
                            )
                            ing.persist_trades([tick])

                tasks.append(asyncio.create_task(_t(sym)))
            elif kind == "bba":
                async def _b(symbol: str) -> None:
                    async for d in adapter.stream_bba(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_bba([data])

                tasks.append(asyncio.create_task(_b(sym)))
            elif kind == "delta":
                async def _d(symbol: str) -> None:
                    async for d in adapter.stream_book_delta(symbol, depth):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_book_delta([data])

                tasks.append(asyncio.create_task(_d(sym)))
            elif kind == "funding":
                async def _f(symbol: str) -> None:
                    async for d in adapter.stream_funding(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_funding([data])

                tasks.append(asyncio.create_task(_f(sym)))
            elif kind in ("oi", "open_interest"):
                async def _oi(symbol: str) -> None:
                    async for d in adapter.stream_open_interest(symbol):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_open_interest([data])

                tasks.append(asyncio.create_task(_oi(sym)))
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


@app.command()
def backfill(
    days: int = typer.Option(1, "--days", help="Number of days to backfill"),
    symbols: List[str] = typer.Option(
        ["BTC/USDT"], "--symbols", help="Symbols to download"
    ),
) -> None:
    """Backfill OHLCV and trades for symbols with rate limiting."""

    setup_logging()
    from ..jobs.backfill import backfill as run_backfill

    asyncio.run(run_backfill(days=days, symbols=symbols))


@app.command("ingest-historical")
def ingest_historical(
    source: str = typer.Argument(..., help="Fuente de datos: kaiko o coinapi"),
    symbol: str = typer.Argument(..., help="Símbolo o par"),
    exchange: str = typer.Option("", help="Exchange para Kaiko"),
    kind: str = typer.Option(
        "trades", help="Tipo de dato: trades, orderbook, open_interest o funding"
    ),
    backend: str = typer.Option("timescale", help="Backend de storage"),
    limit: int = typer.Option(100, help="Límite de trades"),
    depth: int = typer.Option(10, help="Profundidad del order book"),
) -> None:
    """Descargar datos históricos usando Kaiko o CoinAPI."""

    setup_logging()
    if source.lower() == "kaiko":
        from ..connectors.kaiko import KaikoConnector
        from ..data.ingestion import (
            fetch_trades_kaiko,
            fetch_orderbook_kaiko,
            download_kaiko_open_interest,
            download_funding,
        )

        if kind == "orderbook":
            asyncio.run(
                fetch_orderbook_kaiko(
                    exchange, symbol, backend=backend, depth=depth
                )
            )
        elif kind == "open_interest":
            connector = KaikoConnector()
            asyncio.run(
                download_kaiko_open_interest(
                    connector, exchange, symbol, backend=backend, limit=limit
                )
            )
        elif kind == "funding":
            connector = KaikoConnector()
            asyncio.run(download_funding(connector, symbol, backend=backend))
        else:
            asyncio.run(
                fetch_trades_kaiko(
                    exchange, symbol, backend=backend, limit=limit
                )
            )
    elif source.lower() == "coinapi":
        from ..connectors.coinapi import CoinAPIConnector
        from ..data.ingestion import (
            fetch_trades_coinapi,
            fetch_orderbook_coinapi,
            download_coinapi_open_interest,
            download_funding,
        )

        if kind == "orderbook":
            asyncio.run(
                fetch_orderbook_coinapi(
                    symbol, backend=backend, depth=depth
                )
            )
        elif kind == "open_interest":
            connector = CoinAPIConnector()
            asyncio.run(
                download_coinapi_open_interest(
                    connector, symbol, backend=backend, limit=limit
                )
            )
        elif kind == "funding":
            connector = CoinAPIConnector()
            asyncio.run(download_funding(connector, symbol, backend=backend))
        else:
            asyncio.run(
                fetch_trades_coinapi(
                    symbol, backend=backend, limit=limit
                )
            )
    else:  # pragma: no cover - CLI validation
        raise typer.BadParameter("Fuente inválida, usa kaiko o coinapi")


@app.command("run-bot")
def run_bot(
    exchange: str = typer.Option("binance", help="Exchange name"),
    market: str = typer.Option("spot", help="Market type (spot or futures)"),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Trading symbols"),
    testnet: bool = typer.Option(True, help="Use testnet endpoints"),
    trade_qty: float = typer.Option(0.001, help="Order size"),
    leverage: int = typer.Option(1, help="Leverage for futures"),
    dry_run: bool = typer.Option(False, help="Dry run for futures testnet"),
) -> None:
    """Run the live trading bot with configurable exchange and symbols."""

    setup_logging()
    if testnet:
        from ..live.runner_testnet import run_live_testnet

        asyncio.run(
            run_live_testnet(
                exchange=exchange,
                market=market,
                symbols=symbols,
                trade_qty=trade_qty,
                leverage=leverage,
                dry_run=dry_run,
            )
        )
    else:
        from ..live.runner import run_live_binance

        asyncio.run(run_live_binance(symbol=symbols[0]))


@app.command("paper-run")
def paper_run(
    symbol: str = typer.Option("BTC/USDT", "--symbol", help="Trading symbol"),
    strategy: str = typer.Option("breakout_atr", help="Strategy name"),
    metrics_port: int = typer.Option(8000, help="Port to expose metrics"),
) -> None:
    """Run a strategy in paper trading mode with metrics."""

    setup_logging()
    from ..live.runner_paper import run_paper

    asyncio.run(run_paper(symbol=symbol, strategy_name=strategy, metrics_port=metrics_port))


@app.command("real-run")
def real_run(
    exchange: str = typer.Option("binance", help="Exchange name"),
    market: str = typer.Option("spot", help="Market type (spot or futures)"),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Trading symbols"),
    trade_qty: float = typer.Option(0.001, help="Order size"),
    leverage: int = typer.Option(1, help="Leverage for futures"),
    dry_run: bool = typer.Option(False, help="Simulate orders without sending"),
    i_know_what_im_doing: bool = typer.Option(
        False,
        "--i-know-what-im-doing",
        help="Acknowledge that this will trade on a real exchange",
    ),
) -> None:
    """Run the live trading bot on real exchange endpoints."""

    if not i_know_what_im_doing:
        raise typer.BadParameter("pass --i-know-what-im-doing to enable real trading")

    setup_logging()
    from ..live.runner_real import run_live_real

    asyncio.run(
        run_live_real(
            exchange=exchange,
            market=market,
            symbols=symbols,
            trade_qty=trade_qty,
            leverage=leverage,
            dry_run=dry_run,
            i_know_what_im_doing=i_know_what_im_doing,
        )
    )


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
        bal_cfg = getattr(cfg, "balance", {})
        bal_assets = getattr(bal_cfg, "assets", [])
        bal_thr = getattr(bal_cfg, "threshold", 0.0)
        bal_interval = getattr(bal_cfg, "interval", 60.0)
        bal_enabled = getattr(bal_cfg, "enabled", False)

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
            rebalance_interval=bal_interval,
            rebalance_enabled=bal_enabled,
        )

        typer.echo(OmegaConf.to_yaml(cfg))
        asyncio.run(bot.run())

    old = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        _run()
    finally:
        sys.argv = old


@app.command("ingestion-workers")
def run_ingestion_workers(
    config: str = "config/config.yaml",
    backend: str = typer.Option("timescale", help="Storage backend"),
) -> None:
    """Start funding and open-interest ingestion workers defined in a YAML config."""

    import yaml
    from importlib import import_module

    from ..workers import funding_worker, open_interest_worker

    setup_logging()

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    ingestion_cfg = cfg.get("ingestion", {})
    funding_cfg = ingestion_cfg.get("funding", {})
    oi_cfg = ingestion_cfg.get("open_interest", {})

    adapters: dict[str, object] = {}

    def _load_adapter(name: str):
        if name not in adapters:
            module = import_module(f"tradingbot.adapters.{name}")
            cls_name = "".join(part.capitalize() for part in name.split("_")) + "Adapter"
            adapters[name] = getattr(module, cls_name)()
        return adapters[name]

    async def _run() -> None:
        tasks = []
        for exch, symbols in funding_cfg.items():
            adapter = _load_adapter(exch)
            for sym, interval in symbols.items():
                tasks.append(
                    asyncio.create_task(
                        funding_worker(adapter, sym, interval=interval, backend=backend)
                    )
                )
        for exch, symbols in oi_cfg.items():
            adapter = _load_adapter(exch)
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
        from ..backtesting.walk_forward import walk_forward_backtest

        wf_cfg = cfg.walk_forward
        df = walk_forward_backtest(
            wf_cfg.data,
            wf_cfg.symbol,
            wf_cfg.strategy,
            wf_cfg.param_grid,
            train_size=getattr(wf_cfg, "train_size", 1000),
            test_size=getattr(wf_cfg, "test_size", 250),
            latency=getattr(wf_cfg, "latency", 1),
            window=getattr(wf_cfg, "window", 120),
        )

        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / "walk_forward.csv"
        html_path = reports_dir / "walk_forward.html"
        df.to_csv(csv_path, index=False)
        df.to_html(html_path, index=False)

        typer.echo(OmegaConf.to_yaml(cfg))
        typer.echo(df.to_string(index=False))
        typer.echo(f"Reports saved to {csv_path} and {html_path}")

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


@app.command("train-ml")
def train_ml(
    data: str = typer.Argument(..., help="Ruta al CSV con los datos de entrenamiento"),
    target: str = typer.Argument(..., help="Nombre de la columna objetivo"),
    output: str = typer.Argument(..., help="Ruta donde guardar el modelo entrenado"),
) -> None:
    """Entrena un modelo :class:`MLStrategy` y lo guarda en disco."""

    setup_logging()
    import pandas as pd
    from ..strategies.ml_models import MLStrategy

    df = pd.read_csv(data)
    if target not in df.columns:
        raise typer.BadParameter(
            f"Columna objetivo '{target}' no encontrada en {data}"
        )
    y = df[target].to_numpy()
    X = df.drop(columns=[target]).to_numpy()

    strat = MLStrategy()
    strat.train(X, y)
    strat.save_model(output)
    typer.echo(f"Modelo guardado en {output}")


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

