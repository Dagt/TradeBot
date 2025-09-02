"""Backtesting related CLI commands."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from ...logging_conf import setup_logging
from ..utils import _parse_params, _parse_risk_pct, _validate_backtest_venue
from tradingbot.analysis.backtest_report import generate_report

app = typer.Typer(help="Backtesting utilities")


@app.command("cfg-validate")
def cfg_validate(path: str) -> None:
    """Validate a YAML configuration file."""

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
    config: str | None = typer.Option(
        None, "--config", help="YAML config for the strategy"
    ),
    param: list[str] = typer.Option(
        [], "--param", help="Override strategy parameters as key=value pairs"
    ),
    capital: float = typer.Option(0.0, help="Capital inicial"),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk stop loss % (0-1 or 0-100)",
    ),
    fee_bps: float = typer.Option(5.0, "--fee-bps", help="Comisión en bps"),
    slippage_bps: float = typer.Option(1.0, "--slippage-bps", help="Slippage en bps"),
    verbose_fills: bool = typer.Option(
        False, "--verbose-fills", help="Log each fill during backtests"
    ),
    fills_csv: str | None = typer.Option(
        None, "--fills-csv", help="Export fills to CSV"
    ),
) -> dict:
    """Run a simple vectorised backtest from a CSV file."""
    from ...backtest.event_engine import EventDrivenBacktestEngine, SlippageModel
    from ...backtesting.engine import MIN_FILL_QTY
    import pandas as pd

    setup_logging()

    data_path = Path(data)
    df = pd.read_csv(data_path)
    exchange_cfg = {}
    bt_cfg = {}
    min_fill_qty = float(getattr(bt_cfg, "min_fill_qty", MIN_FILL_QTY))
    slippage = None
    params = _parse_params(param) if isinstance(param, list) else {}
    kwargs = dict(params)
    if config is not None:
        kwargs["config_path"] = config
    from ...strategies import STRATEGIES

    strat_cls = STRATEGIES.get(strategy)
    if strat_cls is None:
        raise typer.BadParameter(f"unknown strategy: {strategy}")
    strat = strat_cls(**kwargs)
    eng = EventDrivenBacktestEngine(
        {symbol: df},
        [(strategy, symbol)],
        initial_equity=capital,
        risk_pct=risk_pct,
        verbose_fills=verbose_fills,
        exchange_configs=exchange_cfg,
        min_fill_qty=min_fill_qty,
        slippage=slippage,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    result = eng.run(fills_csv=fills_csv)
    typer.echo(result)
    typer.echo(generate_report(result))
    return result


@app.command("backtest-cfg")
def backtest_cfg(
    config: str,
    capital: float = typer.Option(0.0, help="Capital inicial"),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk stop loss % (0-1 or 0-100)",
    ),
    fee_bps: float = typer.Option(5.0, "--fee-bps", help="Comisión en bps"),
    slippage_bps: float = typer.Option(1.0, "--slippage-bps", help="Slippage en bps"),
    verbose_fills: bool = typer.Option(
        False, "--verbose-fills", help="Log each fill during backtests"
    ),
    fills_csv: str | None = typer.Option(
        None, "--fills-csv", help="Export fills to CSV"
    ),
) -> dict:
    """Run a backtest using a Hydra YAML configuration."""

    import hydra
    from omegaconf import OmegaConf

    setup_logging()
    from ...config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    if fills_csv:
        verbose_fills = False

    @hydra.main(
        config_path=rel_path,
        config_name=cfg_path.stem,
        version_base=None,
    )
    def _run(cfg) -> dict:  # type: ignore[override]
        import pandas as pd

        from ...backtest.event_engine import EventDrivenBacktestEngine, SlippageModel
        from ...backtesting.engine import MIN_FILL_QTY

        data = cfg.backtest.data
        symbol = cfg.backtest.symbol
        strategy = cfg.backtest.strategy

        df = pd.read_csv(data)
        exchange_cfg = OmegaConf.to_container(
            getattr(cfg, "exchange_configs", {}), resolve=True
        )
        bt_cfg = cfg.backtest
        min_fill_qty = float(getattr(bt_cfg, "min_fill_qty", MIN_FILL_QTY))
        slip_cfg = getattr(bt_cfg, "slippage", None)
        slippage = None
        if slip_cfg:
            slip_dict = OmegaConf.to_container(slip_cfg, resolve=True)
            slippage = SlippageModel(**slip_dict)
        eng = EventDrivenBacktestEngine(
            {symbol: df},
            [(strategy, symbol)],
            initial_equity=capital,
            risk_pct=risk_pct,
            verbose_fills=verbose_fills,
            exchange_configs=exchange_cfg,
            min_fill_qty=min_fill_qty,
            slippage=slippage,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        params = {}
        from ...strategies import STRATEGIES

        strat_cls = STRATEGIES.get(strategy)
        if strat_cls is None:
            raise typer.BadParameter(f"unknown strategy: {strategy}")
        strat = strat_cls(**params)
        eng.strategies[(strategy, symbol)] = strat
        result = eng.run(fills_csv=fills_csv)
        typer.echo(OmegaConf.to_yaml(cfg))
        typer.echo(result)
        typer.echo(generate_report(result))
        return result

    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        result = _run()
    finally:
        sys.argv = old_argv
    return result


@app.command("backtest-db")
def backtest_db(
    venue: str = typer.Option(
        "binance_spot",
        "--venue",
        callback=_validate_backtest_venue,
        help="Trading venue",
    ),
    symbol: str = typer.Option("BTC/USDT", "--symbol"),
    strategy: str = typer.Option("breakout_atr", help="Strategy name"),
    config: str | None = typer.Option(
        None, "--config", help="YAML config for the strategy"
    ),
    param: list[str] = typer.Option(
        [], "--param", help="Override strategy parameters as key=value pairs"
    ),
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
    timeframe: str = typer.Option("3m", help="Bar timeframe"),
    capital: float = typer.Option(0.0, help="Capital inicial"),
    risk_pct: float = typer.Option(
        0.0,
        "--risk-pct",
        callback=_parse_risk_pct,
        help="Risk stop loss % (0-1 or 0-100)",
    ),
    fee_bps: float = typer.Option(5.0, "--fee-bps", help="Comisión en bps"),
    slippage_bps: float = typer.Option(1.0, "--slippage-bps", help="Slippage en bps"),
    verbose_fills: bool = typer.Option(
        False, "--verbose-fills", help="Log each fill during backtests"
    ),
    fills_csv: str | None = typer.Option(
        None, "--fills-csv", help="Export fills to CSV"
    ),
) -> None:
    """Run a backtest using data stored in the database."""

    from datetime import datetime
    import pandas as pd
    from omegaconf import OmegaConf
    from ...storage.timescale import get_engine, select_bars
    from ...backtest.event_engine import EventDrivenBacktestEngine, SlippageModel
    from ...backtesting.engine import MIN_FILL_QTY
    from ...config.hydra_conf import load_config
    from ...core import normalize

    setup_logging()
    symbol = normalize(symbol)
    timeframe = timeframe.lower()
    engine = get_engine()
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        rows = select_bars(
            engine,
            exchange=venue,
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            timeframe=timeframe,
        )
        if not rows:
            typer.echo(f"no data for {symbol}")
            raise typer.Exit()
        df = (
            pd.DataFrame(rows)
            .astype({"o": float, "h": float, "l": float, "c": float, "v": float})
            .rename(
                columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            )
            .set_index("ts")
        )
        if not isinstance(fills_csv, str):
            fills_csv = None
        if fills_csv:
            verbose_fills = False
        cfg_all = load_config()
        exchange_cfg_all = OmegaConf.to_container(
            getattr(cfg_all, "exchange_configs", {}), resolve=True
        )
        bt_cfg = getattr(cfg_all, "backtest", {})
        min_fill_qty = float(getattr(bt_cfg, "min_fill_qty", MIN_FILL_QTY))
        slip_cfg = getattr(bt_cfg, "slippage", None)
        slippage = None
        if slip_cfg:
            slip_dict = OmegaConf.to_container(slip_cfg, resolve=True)
            slippage = SlippageModel(**slip_dict)
        venue_cfg = exchange_cfg_all.get(venue, {})
        if not venue_cfg:
            typer.echo(f"missing config for {venue}")
            raise typer.Exit()
        exchange_cfg = {venue: venue_cfg}
        eng = EventDrivenBacktestEngine(
            {symbol: df},
            [(strategy, symbol, venue)],
            initial_equity=capital,
            risk_pct=risk_pct,
            verbose_fills=verbose_fills,
            exchange_configs=exchange_cfg,
            min_fill_qty=min_fill_qty,
            slippage=slippage,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        params = _parse_params(param) if isinstance(param, list) else {}
        if not isinstance(config, str):
            config = None
        from ...strategies import STRATEGIES

        strat_cls = STRATEGIES.get(strategy)
        if strat_cls is None:
            raise typer.BadParameter(f"unknown strategy: {strategy}")
        kwargs = dict(params)
        if config is not None:
            kwargs["config_path"] = config
        strat = strat_cls(**kwargs)
        eng.strategies[(strategy, symbol)] = strat
        result = eng.run(fills_csv=fills_csv)
        typer.echo(result)
        typer.echo(generate_report(result))
    finally:
        engine.dispose()


@app.command("walk-forward")
def walk_forward_cfg(
    config: str,
    reports_dir: str = typer.Option("reports", help="Directory for reports"),
) -> None:
    """Run walk-forward optimization from a Hydra configuration."""

    import hydra
    from omegaconf import OmegaConf
    import pandas as pd

    setup_logging()
    from ...config import hydra_conf as _  # noqa: F401

    cfg_path = Path(config)
    rel_path = os.path.relpath(cfg_path.parent, Path(__file__).parent)

    @hydra.main(config_path=rel_path, config_name=cfg_path.stem, version_base=None)
    def _run(cfg) -> None:  # type: ignore[override]
        from ...backtesting.walk_forward import walk_forward_backtest

        wf_cfg = cfg.walk_forward
        df = walk_forward_backtest(
            data_path=wf_cfg.data,
            symbol=wf_cfg.symbol,
            strategy_name=wf_cfg.strategy,
            param_grid=OmegaConf.to_container(wf_cfg.param_grid, resolve=True),
        )
        reports = Path(reports_dir)
        reports.mkdir(parents=True, exist_ok=True)
        csv_path = reports / "walk_forward.csv"
        html_path = reports / "walk_forward.html"
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
    sys.exit(0)


@app.command("train-ml")
def train_ml(
    data: str = typer.Argument(..., help="Ruta al CSV con los datos de entrenamiento"),
    target: str = typer.Argument(..., help="Nombre de la columna objetivo"),
    output: str = typer.Argument(..., help="Ruta donde guardar el modelo entrenado"),
) -> None:
    """Entrena un modelo MLStrategy y lo guarda en disco."""

    setup_logging()
    import pandas as pd
    from ...strategies.ml_models import MLStrategy

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

