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
import inspect
import ast
import textwrap
from typing import List

import typer

from .. import adapters
from ..adapters import (
    BinanceFuturesAdapter,
    BinanceSpotAdapter,
    BinanceFuturesWSAdapter,
    BinanceSpotWSAdapter,
    BybitFuturesAdapter,
    BybitSpotAdapter,
    BybitWSAdapter,
    DeribitAdapter,
    DeribitWSAdapter,
    OKXFuturesAdapter,
    OKXSpotAdapter,
    OKXWSAdapter,
)
from ..logging_conf import setup_logging
from tradingbot.analysis.backtest_report import generate_report
from tradingbot.core.symbols import normalize
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


# Manual mapping of venue names to adapter classes to avoid relying on
# capitalization conventions. This ensures acronyms such as OKX resolve
# correctly without deriving the class name dynamically.
_ADAPTER_CLASS_MAP: dict[str, type[adapters.ExchangeAdapter]] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "binance_spot_ws": BinanceSpotWSAdapter,
    "binance_futures_ws": BinanceFuturesWSAdapter,
    "bybit_spot": BybitSpotAdapter,
    "bybit_futures": BybitFuturesAdapter,
    "bybit_futures_ws": BybitWSAdapter,
    "okx_spot": OKXSpotAdapter,
    "okx_futures": OKXFuturesAdapter,
    "okx_futures_ws": OKXWSAdapter,
    "deribit_futures": DeribitAdapter,
    "deribit_futures_ws": DeribitWSAdapter,
}


def get_adapter_class(name: str) -> type[adapters.ExchangeAdapter] | None:
    """Return the adapter class for ``name`` if available."""

    return _ADAPTER_CLASS_MAP.get(name)


def get_supported_kinds(adapter_cls: type[adapters.ExchangeAdapter]) -> list[str]:
    """Return a sorted list of stream kinds supported by ``adapter_cls``.

    The function inspects ``adapter_cls`` for methods named ``stream_*`` and
    returns the suffixes normalised to match CLI ``kind`` parameters.  Methods
    inherited directly from :class:`~tradingbot.adapters.ExchangeAdapter` that
    are not overridden are ignored.
    """

    kinds_attr = getattr(adapter_cls, "supported_kinds", None)
    if kinds_attr:
        return sorted(kinds_attr)

    kinds: set[str] = set()
    for name in dir(adapter_cls):
        if not name.startswith("stream_"):
            continue
        attr = getattr(adapter_cls, name)
        if not callable(attr):
            continue
        base_attr = getattr(adapters.ExchangeAdapter, name, None)
        if base_attr is attr:
            # method not implemented by subclass
            continue
        try:
            src = inspect.getsource(attr)
        except OSError:  # pragma: no cover - builtins or C extensions
            src = ""
        if src:
            fn_node = ast.parse(textwrap.dedent(src)).body[0]
            body = getattr(fn_node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                exc = body[0].exc
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    continue
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "NotImplementedError"
                ):
                    continue
        kind = name[len("stream_"):]
        if kind in ("order_book", "orderbook"):
            kind = "orderbook"
        elif kind == "book_delta":
            kind = "delta"
        kinds.add(kind)
    name = getattr(adapter_cls, "name", "")
    if "futures" not in name:
        kinds.discard("funding")
        kinds.discard("open_interest")
    return sorted(kinds)


def _get_available_venues() -> set[str]:
    """Return venue names available for the CLI."""

    return set(_ADAPTER_CLASS_MAP)


_AVAILABLE_VENUES = _get_available_venues()


@app.command()
def ingest(
    venue: str = typer.Option(
        "binance_spot", help=f"Data venue adapter ({', '.join(sorted(_AVAILABLE_VENUES))})"
    ),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Market symbols"),
    depth: int = typer.Option(10, help="Order book depth"),
    kind: str = typer.Option(
        "orderbook",
        "--kind",
        help="Data kind: trades,trades_multi,orderbook,bba,delta,funding,oi",
    ),
    persist: bool = typer.Option(False, "--persist", help="Persist data"),
    backend: str = typer.Option("timescale", "--backend"),
) -> None:
    """Stream market data from a venue and optionally persist it."""

    setup_logging()

    from ..bus import EventBus
    from ..data import ingestion as ing
    from ..types import Tick
    from ..storage import quest as qs_storage, timescale as ts_storage

    if venue not in _AVAILABLE_VENUES:
        choices = ", ".join(sorted(_AVAILABLE_VENUES))
        raise typer.BadParameter(f"Invalid venue, choose one of: {choices}")

    adapter_class = get_adapter_class(venue)
    if adapter_class is None:
        choices = ", ".join(sorted(_AVAILABLE_VENUES))
        raise typer.BadParameter(f"Invalid venue, choose one of: {choices}")
    adapter = adapter_class()

    alias_kind = "open_interest" if kind == "oi" else kind
    supported_kinds = get_supported_kinds(adapter_class)
    if alias_kind not in supported_kinds:
        choices = ", ".join(sorted(supported_kinds))
        raise typer.BadParameter(
            f"adapter does not support {alias_kind} (supported: {choices})"
        )
    kind = alias_kind

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
                            tick = Tick(
                                ts=d.get("ts"),
                                exchange=adapter.name,
                                symbol=symbol,
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
                            tick = Tick(
                                ts=d.get("ts"),
                                exchange=adapter.name,
                                symbol=d.get("symbol"),
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
                            data.update({"exchange": adapter.name, "symbol": symbol})
                            ing.persist_bba([data], backend=backend)

                tasks.append(asyncio.create_task(_b(sym)))
            elif kind == "delta":
                async def _d(symbol: str) -> None:
                    async for d in adapter.stream_book_delta(symbol, depth):
                        typer.echo(str(d))
                        if persist:
                            data = dict(d)
                            data.update({"exchange": adapter.name, "symbol": symbol})
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
    start: str | None = typer.Option(
        None, "--start", help="Start datetime in ISO format"
    ),
    end: str | None = typer.Option(
        None, "--end", help="End datetime in ISO format"
    ),
) -> None:
    """Backfill OHLCV and trades for symbols with rate limiting."""

    setup_logging()
    from datetime import datetime, timezone
    from ..jobs.backfill import backfill as run_backfill

    def _parse(dt: str | None) -> datetime | None:
        if dt is None:
            return None
        parsed = datetime.fromisoformat(dt)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    asyncio.run(
        run_backfill(
            days=days,
            symbols=symbols,
            start=_parse(start),
            end=_parse(end),
        )
    )


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
    stop_loss: float = typer.Option(0.0, "--stop-loss", help="Strategy stop loss percentage"),
    take_profit: float = typer.Option(0.0, "--take-profit", help="Strategy take profit percentage"),
    stop_loss_pct: float = typer.Option(0.0, "--stop-loss-pct", help="Risk manager stop loss percentage"),
    max_drawdown_pct: float = typer.Option(0.0, "--max-drawdown-pct", help="Risk manager max drawdown percentage"),
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
    config: str | None = typer.Option(None, "--config", help="YAML config for the strategy"),
) -> None:
    """Run a strategy in paper trading mode with metrics."""

    setup_logging()
    from ..live.runner_paper import run_paper

    asyncio.run(
        run_paper(
            symbol=symbol,
            strategy_name=strategy,
            config_path=config,
            metrics_port=metrics_port,
        )
    )


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

    from ..workers import funding_worker, open_interest_worker

    setup_logging()

    with open(config) as fh:
        cfg = yaml.safe_load(fh)

    ingestion_cfg = cfg.get("ingestion", {})
    funding_cfg = ingestion_cfg.get("funding", {})
    oi_cfg = ingestion_cfg.get("open_interest", {})

    adapters_cache: dict[str, object] = {}

    def _load_adapter(name: str):
        if name not in adapters_cache:
            cls = get_adapter_class(name)
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




@app.command("cfg-validate")
def cfg_validate(path: str) -> None:
    """Validate a YAML configuration file.

    Ensures required fields exist for known sections such as ``backtest`` and
    ``walk_forward``.
    """

    import yaml

    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}

    required = {
        "backtest": ["data", "symbol", "strategy"],
        "walk_forward": ["data", "symbol", "strategy", "param_grid"],
    }
    missing: list[str] = []
    for section, keys in required.items():
        section_cfg = cfg.get(section, {}) or {}
        for key in keys:
            if key not in section_cfg:
                missing.append(f"{section}.{key}")
    if missing:
        raise typer.BadParameter("Missing required fields: " + ", ".join(missing))

    typer.echo("Configuration valid")


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


@app.command("backtest-db")
def backtest_db(
    exchange: str = typer.Option("binance", help="Exchange name"),
    symbol: str = typer.Option(..., "--symbol", help="Trading symbol"),
    strategy: str = typer.Option("breakout_atr", help="Strategy name"),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    timeframe: str = typer.Option("1m", help="Bar timeframe"),
) -> None:
    """Run a backtest using data stored in the database."""

    from datetime import datetime
    import pandas as pd
    from ..storage.timescale import get_engine, select_bars
    from ..backtest.event_engine import EventDrivenBacktestEngine

    setup_logging()
    engine = get_engine()
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    rows = select_bars(
        engine,
        exchange=exchange,
        symbol=symbol,
        start=start_dt,
        end=end_dt,
        timeframe=timeframe,
    )
    if not rows:
        typer.echo("no data")
        raise typer.Exit()
    df = (
        pd.DataFrame(rows)
        .rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        .set_index("ts")
    )
    eng = EventDrivenBacktestEngine({symbol: df}, [(strategy, symbol)])
    result = eng.run()
    typer.echo(result)
    typer.echo(generate_report(result))


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

