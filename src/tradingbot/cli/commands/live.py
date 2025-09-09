"""Live trading related CLI commands."""
from __future__ import annotations

import asyncio
from typing import List

import os
import sys
import typer

from ...logging_conf import setup_logging
from ..utils import _parse_params, _parse_risk_pct, _validate_venue

app = typer.Typer(help="Live trading utilities")


@app.command("run-bot")
def run_bot(
    strategy: str = typer.Option("breakout_atr", "--strategy", help="Strategy name"),
    venue: str = typer.Option(
        "binance_spot",
        "--venue",
        callback=_validate_venue,
        help="Trading venue",
    ),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Trading symbols"),
    testnet: bool = typer.Option(True, help="Use testnet endpoints"),
    leverage: int = typer.Option(1, help="Leverage for futures"),
    dry_run: bool = typer.Option(False, help="Dry run for futures testnet"),
    timeframe: str = typer.Option("1m", "--timeframe", help="Bar timeframe (e.g., 1m,5m,15m)"),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk manager loss percentage (0-1 or 0-100)",
    ),
    daily_max_loss_pct: float = typer.Option(
        0.05, "--daily-max-loss-pct", help="Daily loss limit as fraction of equity"
    ),
    daily_max_drawdown_pct: float = typer.Option(
        0.05, "--daily-max-drawdown-pct", help="Intraday max drawdown limit"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level (e.g., INFO, DEBUG)"
    ),
    config: str | None = typer.Option(None, "--config", help="YAML config for the strategy"),
    param: list[str] = typer.Option(
        [], "--param", help="Override strategy parameters as key=value pairs"
    ),
) -> None:
    """Run the live trading bot with configurable venue and symbols."""

    setup_logging(log_level)
    params = _parse_params(param) if isinstance(param, list) else {}
    from ...core.account import Account
    from ...risk.portfolio_guard import PortfolioGuard, GuardConfig
    from ...risk.service import RiskService

    _guard = PortfolioGuard(GuardConfig(venue=venue))
    RiskService(_guard, account=Account(float("inf")), risk_pct=risk_pct)
    if testnet:
        from ...live.runner_testnet import run_live_testnet

        exchange, market = venue.split("_", 1)
        asyncio.run(
            run_live_testnet(
                exchange=exchange,
                market=market,
                symbols=symbols,
                risk_pct=risk_pct,
                leverage=leverage,
                dry_run=dry_run,
                daily_max_loss_pct=daily_max_loss_pct,
                daily_max_drawdown_pct=daily_max_drawdown_pct,
                strategy_name=strategy,
                config_path=config,
                params=params,
                timeframe=timeframe,
            )
        )
    else:
        from ...live.runner import run_live_binance

        asyncio.run(
            run_live_binance(
                symbol=symbols[0],
                risk_pct=risk_pct,
                daily_max_loss_pct=daily_max_loss_pct,
                daily_max_drawdown_pct=daily_max_drawdown_pct,
                strategy_name=strategy,
                config_path=config,
                params=params,
                timeframe=timeframe,
            )
        )


@app.command("paper-run")
def paper_run(
    symbol: str = typer.Option("BTC/USDT", "--symbol", help="Trading symbol"),
    strategy: str = typer.Option("breakout_atr", help="Strategy name"),
    metrics_port: int = typer.Option(8000, help="Port to expose metrics"),
    config: str | None = typer.Option(None, "--config", help="YAML config for the strategy"),
    param: list[str] = typer.Option(
        [], "--param", help="Override strategy parameters as key=value pairs"
    ),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk manager loss percentage (0-1 or 0-100)",
    ),
    timeframe: str = typer.Option("1m", "--timeframe", help="Bar timeframe (e.g., 1m,5m,15m)"),
    initial_cash: float = typer.Option(
        1000.0, "--initial-cash", help="Initial cash for paper trading"
    ),
) -> None:
    """Run a strategy in paper trading mode with metrics."""

    setup_logging()
    from ...live.runner_paper import run_paper

    from ...core.account import Account
    from ...risk.portfolio_guard import PortfolioGuard, GuardConfig
    from ...risk.service import RiskService

    _guard = PortfolioGuard(GuardConfig(venue="paper"))
    RiskService(_guard, account=Account(float("inf")), risk_pct=risk_pct)

    params = _parse_params(param)

    asyncio.run(
        run_paper(
            symbol=symbol,
            strategy_name=strategy,
            config_path=config,
            metrics_port=metrics_port,
            risk_pct=risk_pct,
            params=params,
            timeframe=timeframe,
            initial_cash=initial_cash,
        )
    )


@app.command("real-run")
def real_run(
    venue: str = typer.Option(
        "binance_spot",
        "--venue",
        callback=_validate_venue,
        help="Trading venue",
    ),
    symbols: List[str] = typer.Option(["BTC/USDT"], "--symbol", help="Trading symbols"),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk manager loss percentage (0-1 or 0-100)",
    ),
    leverage: int = typer.Option(1, help="Leverage for futures"),
    dry_run: bool = typer.Option(False, help="Simulate orders without sending"),
    daily_max_loss_pct: float = typer.Option(
        0.05, "--daily-max-loss-pct", help="Daily loss limit as fraction of equity"
    ),
    daily_max_drawdown_pct: float = typer.Option(
        0.05, "--daily-max-drawdown-pct", help="Intraday max drawdown limit"
    ),
    timeframe: str = typer.Option("1m", "--timeframe", help="Bar timeframe (e.g., 1m,5m,15m)"),
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
    from ...live.runner_real import run_live_real

    from ...core.account import Account
    from ...risk.portfolio_guard import PortfolioGuard, GuardConfig
    from ...risk.service import RiskService

    exchange, market = venue.split("_", 1)

    _guard = PortfolioGuard(GuardConfig(venue=venue))
    RiskService(_guard, account=Account(float("inf")), risk_pct=risk_pct)

    asyncio.run(
        run_live_real(
            exchange=exchange,
            market=market,
            symbols=symbols,
            risk_pct=risk_pct,
            leverage=leverage,
            dry_run=dry_run,
            daily_max_loss_pct=daily_max_loss_pct,
            daily_max_drawdown_pct=daily_max_drawdown_pct,
            timeframe=timeframe,
            i_know_what_im_doing=i_know_what_im_doing,
        )
    )


@app.command("daemon")
def run_daemon(config: str = "config/config.yaml") -> None:
    """Launch the TradeBot daemon using a Hydra configuration."""

    from pathlib import Path

    import hydra
    from omegaconf import OmegaConf

    setup_logging()

    # Register dataclasses and Hydra config
    from ...config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    @hydra.main(config_path=rel_path, config_name=cfg_path.stem, version_base=None)
    def _run(cfg) -> None:  # type: ignore[override]
        from ...adapters.binance_spot_ws import BinanceSpotWSAdapter
        from ...bus import EventBus
        from ...live.daemon import TradeBotDaemon
        from ...risk.portfolio_guard import PortfolioGuard, GuardConfig
        from ...core.account import Account
        from ...risk.service import RiskService
        from ...strategies.breakout_atr import BreakoutATR
        from ...execution.router import ExecutionRouter

        adapter = BinanceSpotWSAdapter()
        bus = EventBus()
        risk = RiskService(PortfolioGuard(GuardConfig(venue="cli")), account=Account(float("inf")), bus=bus, risk_pct=0.0)
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


@app.command("tri-arb")
def tri_arb(
    route: str = typer.Argument(..., help="Ruta BASE-MID-QUOTE, ej. BTC-ETH-USDT"),
) -> None:
    """Ejecutar arbitrage triangular simple en Binance."""

    setup_logging()
    from ...live.runner_triangular import TriConfig, run_triangular_binance
    from ...strategies.arbitrage_triangular import TriRoute

    try:
        base, mid, quote = route.split("-")
    except ValueError as exc:  # pragma: no cover - validated by typer
        raise typer.BadParameter("Formato de ruta inválido, usa BASE-MID-QUOTE") from exc

    cfg = TriConfig(route=TriRoute(base, mid, quote))
    asyncio.run(run_triangular_binance(cfg))


@app.command("cross-arb")
def cross_arb(
    symbol: str = typer.Argument("BTC/USDT", help="Símbolo a arbitrar"),
    spot: str = typer.Argument(..., help="Adapter spot, ej. binance_spot"),
    perp: str = typer.Argument(..., help="Adapter perp, ej. binance_futures"),
    threshold: float = typer.Option(0.001, help="Umbral de premium (decimales)"),
) -> None:
    """Arbitraje entre spot y perp usando dos adapters."""

    setup_logging()
    from ...adapters import (
        BinanceFuturesAdapter,
        BinanceSpotAdapter,
        BybitFuturesAdapter,
        BybitSpotAdapter,
        OKXFuturesAdapter,
        OKXSpotAdapter,
    )
    from ...strategies.cross_exchange_arbitrage import (
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
    )
    asyncio.run(run_cross_exchange_arbitrage(cfg))


@app.command("run-cross-arb")
def run_cross_arb(
    symbol: str = typer.Argument("BTC/USDT", help="Símbolo a arbitrar"),
    spot: str = typer.Argument(..., help="Adapter spot, ej. binance_spot"),
    perp: str = typer.Argument(..., help="Adapter perp, ej. binance_futures"),
    threshold: float = typer.Option(0.001, help="Umbral de premium (decimales)"),
) -> None:
    """Ejecuta el runner de arbitraje spot/perp con ExecutionRouter."""

    setup_logging()
    from ...adapters import (
        BinanceFuturesAdapter,
        BinanceSpotAdapter,
        BybitFuturesAdapter,
        BybitSpotAdapter,
        OKXFuturesAdapter,
        OKXSpotAdapter,
    )
    from ...strategies.cross_exchange_arbitrage import CrossArbConfig
    from ...live.runner_cross_exchange import run_cross_exchange

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
    )
    asyncio.run(run_cross_exchange(cfg))

