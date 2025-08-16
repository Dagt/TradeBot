# src/tradingbot/cli/main.py
import typer, logging, time
import pandas as pd

from ..logging_conf import setup_logging
from ..backtest.event_engine import run_backtest_csv, run_backtest_mlflow
from ..strategies import STRATEGIES
from ..risk.manager import RiskManager
from ..execution.paper import PaperAdapter
from ..execution.order_types import Order
from ..execution.router import ExecutionRouter
from ..execution.algos import TWAP, VWAP, POV
from ..live.runner import run_live_binance
from ..live.runner_futures_testnet import run_live_binance_futures_testnet
from ..live.runner_triangular import run_triangular_binance, TriConfig, TriRoute
from ..market.exchange_meta import ExchangeMeta
from ..execution.normalize import adjust_order
from ..live.runner_spot_testnet import run_live_binance_spot_testnet
from ..live.runner_spot_testnet_multi import run_live_binance_spot_testnet_multi
from ..live.runner_futures_testnet_multi import run_live_binance_futures_testnet_multi

app = typer.Typer(add_completion=False)

@app.command()
def backtest(data: str, symbol: str = "BTC/USDT", strategy: str = "breakout_atr"):
    """Backtest vectorizado simple desde CSV (columnas: timestamp, open, high, low, close, volume)"""
    setup_logging()
    res = run_backtest_csv({symbol: data}, [(strategy, symbol)])
    typer.echo(res)


@app.command()
def backtest_cfg(cfg_path: str):
    """Backtest usando configuración YAML con Hydra."""
    setup_logging()
    from pathlib import Path
    from hydra import compose, initialize

    cfg_file = Path(cfg_path)
    with initialize(version_base=None, config_path=str(cfg_file.parent)):
        cfg = compose(config_name=cfg_file.stem)

    csv_paths = {
        sym: str((cfg_file.parent / Path(path)).resolve())
        for sym, path in cfg.csv_paths.items()
    }
    strategies = [tuple(item) for item in cfg.strategies]
    latency = int(cfg.get("latency", 1))
    window = int(cfg.get("window", 120))

    if getattr(cfg, "mlflow", None):
        run_name = cfg.mlflow.get("run_name", "backtest")
        res = run_backtest_mlflow(
            csv_paths,
            strategies,
            latency=latency,
            window=window,
            run_name=run_name,
        )
    else:
        res = run_backtest_csv(
            csv_paths,
            strategies,
            latency=latency,
            window=window,
        )
    typer.echo(res)

@app.command()
def run_csv_paper(
    data: str,
    symbol: str = "BTC/USDT",
    strategy: str = "breakout_atr",
    sleep_ms: int = 50,
    max_bars: int = 0,
):
    """
    Reproduce un feed con CSV y ejecuta órdenes en un broker de papel.
    Útil para ver la tubería Estrategia -> Riesgo -> Broker (simulada).
    """
    setup_logging()
    log = logging.getLogger("cli.run_csv_paper")

    df = pd.read_csv(data)
    needed_cols = {"timestamp","open","high","low","close","volume"}
    if not needed_cols.issubset(set(df.columns)):
        raise SystemExit(f"CSV debe contener columnas: {sorted(needed_cols)}")

    strat_cls = STRATEGIES.get(strategy)
    if strat_cls is None:
        raise SystemExit(f"estrategia desconocida: {strategy}")
    strat = strat_cls()
    risk = RiskManager(max_pos=1.0)
    broker = PaperAdapter(fee_bps=1.5)  # 1.5 bps = 0.015%

    window = 120
    fills = []
    bars_done = 0

    for i in range(len(df)):
        if max_bars and bars_done >= max_bars:
            break
        if i < window:
            broker.update_last_price(symbol, float(df["close"].iloc[i]))
            continue

        win = df.iloc[i-window:i].copy()
        bar = {"window": win}
        last_close = float(df["close"].iloc[i])
        broker.update_last_price(symbol, last_close)

        sig = strat.on_bar(bar)
        if sig is None:
            continue

        delta = risk.size(sig.side, sig.strength)
        if abs(delta) > 1e-9:
            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            import asyncio
            resp = asyncio.run(broker.place_order(symbol, side, "market", qty))
            fills.append(resp)
            log.info("FILL %s", resp)

        bars_done += 1
        time.sleep(sleep_ms / 1000.0)

    eq = broker.equity()
    typer.echo({"fills": len(fills), "equity": eq, "last_px": float(df['close'].iloc[-1])})


@app.command()
def demo_algos(symbol: str = "BTC/USDT", qty: float = 1.0):
    """Demuestra TWAP, VWAP y POV usando el broker de papel."""
    setup_logging()
    import asyncio

    async def _run():
        broker = PaperAdapter()
        broker.update_last_price(symbol, 100.0)
        router = ExecutionRouter(broker)
        order = Order(symbol=symbol, side="buy", type_="market", qty=qty)
        twap = await TWAP(router, slices=4).execute(order)

        broker2 = PaperAdapter()
        broker2.update_last_price(symbol, 100.0)
        router2 = ExecutionRouter(broker2)
        order2 = Order(symbol=symbol, side="buy", type_="market", qty=qty)
        vwap = await VWAP(router2, volumes=[1, 2, 1]).execute(order2)

        broker3 = PaperAdapter()
        broker3.update_last_price(symbol, 100.0)
        router3 = ExecutionRouter(broker3)
        order3 = Order(symbol=symbol, side="buy", type_="market", qty=qty)

        async def trades():
            for _ in range(8):
                yield {"ts": 0, "price": 100.0, "qty": qty, "side": "buy"}

        pov = await POV(router3, participation_rate=0.5).execute(order3, trades())
        return {"twap": len(twap), "vwap": len(vwap), "pov": len(pov)}

    res = asyncio.run(_run())
    typer.echo(res)

@app.command()
def run_live_binance_cli(symbol: str = "BTC/USDT", fee_bps: float = 1.5, persist_pg: bool = False):
    """
    Conecta WS de Binance y ejecuta la estrategia en vivo con broker de papel.
    Ctrl+C para salir.
    """
    setup_logging()
    import asyncio
    try:
        asyncio.run(run_live_binance(symbol=symbol, fee_bps=fee_bps, persist_pg=persist_pg))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

@app.command()
def run_live_binance_futures_testnet_cli(
    symbol: str = "BTC/USDT",
    leverage: int = 5,
    trade_qty: float = 0.001,
    dry_run: bool = True
):
    """
    Conecta WS (precio) + ejecuta órdenes en Binance Futures TESTNET (o Paper si dry_run=True).
    """
    setup_logging()
    import asyncio
    try:
        asyncio.run(run_live_binance_futures_testnet(
            symbol=symbol,
            leverage=leverage,
            trade_qty=trade_qty,
            dry_run=dry_run
        ))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

@app.command()
def run_triangular_binance_cli(
    base: str = "BTC",
    mid: str = "ETH",
    quote: str = "USDT",
    notional_quote: float = 50.0,
    taker_fee_bps: float = 7.5,
    buffer_bps: float = 3.0,
    edge_threshold_bps: float = 10.0,  # 0.10% por default
    persist_pg: bool = False
):
    """
    Arbitraje triangular intra-exchange con WS público (Binance) y ejecución PAPER.
    Si --persist-pg true, guarda señales y órdenes en Timescale.
    """
    setup_logging()
    cfg = TriConfig(
        route=TriRoute(base=base, mid=mid, quote=quote),
        taker_fee_bps=taker_fee_bps,
        buffer_bps=buffer_bps,
        notional_quote=notional_quote,
        edge_threshold=edge_threshold_bps/10000.0,
        persist_pg=persist_pg
    )
    import asyncio
    try:
        asyncio.run(run_triangular_binance(cfg))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

@app.command()
def validate_order_cli(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    type_: str = "market",
    qty: float = 0.001,
    price: float = 0.0,
    mark_price: float = 65000.0
):
    """
    Valida/normaliza una orden contra filtros del exchange (tick/step/minNotional).
    Útil antes de operar real.
    """
    setup_logging()
    import json

    meta = ExchangeMeta.binanceusdm_testnet()
    meta.load_markets()
    rules = meta.rules_for(symbol)

    px = None if type_.lower()=="market" else float(price)
    ar = adjust_order(px, float(qty), float(mark_price), rules, side)
    result = {
        "symbol": symbol,
        "side": side,
        "type": type_.lower(),
        "input_qty": qty,
        "input_price": None if px is None else price,
        "rules": {
            "price_step": rules.price_step,
            "qty_step": rules.qty_step,
            "min_qty": rules.min_qty,
            "min_notional": rules.min_notional
        },
        "adjusted": {
            "price": ar.price,
            "qty": ar.qty,
            "notional": ar.notional,
            "ok": ar.ok,
            "reason": ar.reason
        }
    }
    typer.echo(json.dumps(result, indent=2))

@app.command()
def run_live_binance_spot_testnet_cli(
    symbol: str = "BTC/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False
):
    """
    Ejecuta la estrategia BreakoutATR en Binance SPOT Testnet con validación de tamaños.
    """
    setup_logging()
    import asyncio
    try:
        asyncio.run(run_live_binance_spot_testnet(symbol=symbol, trade_qty=trade_qty, persist_pg=persist_pg))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

# --- comando: validar orden spot ---
@app.command()
def validate_order_spot_cli(
    symbol: str = "BTC/USDT",
    side: str = "buy",
    type_: str = "market",
    qty: float = 0.001,
    price: float = 0.0,
    mark_price: float = 65000.0
):
    """
    Valida/normaliza una orden contra filtros SPOT testnet (tick/step/minNotional).
    """
    setup_logging()
    import json
    meta = ExchangeMeta.binance_spot_testnet()
    meta.load_markets()
    rules = meta.rules_for(symbol)
    px = None if type_.lower()=="market" else float(price)
    ar = adjust_order(px, float(qty), float(mark_price), rules, side)
    result = {
        "venue": "binance_spot_testnet",
        "symbol": symbol,
        "side": side,
        "type": type_.lower(),
        "input_qty": qty,
        "input_price": None if px is None else price,
        "rules": {
            "price_step": rules.price_step,
            "qty_step": rules.qty_step,
            "min_qty": rules.min_qty,
            "min_notional": rules.min_notional
        },
        "adjusted": {
            "price": ar.price,
            "qty": ar.qty,
            "notional": ar.notional,
            "ok": ar.ok,
            "reason": ar.reason
        }
    }
    typer.echo(json.dumps(result, indent=2))

# Spot balances
@app.command()
def balances_spot_cli():
    """Muestra balances SPOT testnet (free base/quote de algunas monedas comunes)."""
    from ..market.exchange_meta import ExchangeMeta
    import json, asyncio
    async def run():
        meta = ExchangeMeta.binance_spot_testnet()
        await asyncio.to_thread(meta.client.load_markets, True)
        bal = await asyncio.to_thread(meta.client.fetch_balance)
        keep = ["USDT","BTC","ETH","BNB"]
        out = {k: bal.get(k) for k in keep if bal.get(k)}
        print(json.dumps(out, indent=2))
    import asyncio; asyncio.run(run())

# Futures balances
@app.command()
def balances_futures_cli():
    """Muestra USDT libre en Futures testnet."""
    from ..market.exchange_meta import ExchangeMeta
    import json, asyncio
    async def run():
        meta = ExchangeMeta.binanceusdm_testnet()
        await asyncio.to_thread(meta.client.load_markets, True)
        bal = await asyncio.to_thread(meta.client.fetch_balance, {"type":"future"})
        out = {"USDT": bal.get("USDT")}
        print(json.dumps(out, indent=2))
    import asyncio; asyncio.run(run())

# --- comando multi-símbolo SPOT TESTNET ---
@app.command()
def run_live_binance_spot_testnet_multi_cli(
    symbols: str = "BTC/USDT,ETH/USDT",
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 1000.0,
    per_symbol_cap_usdt: float = 500.0,
    auto_close: bool = True,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30
):
    setup_logging()
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    import asyncio
    try:
        from ..live.runner_spot_testnet_multi import run_live_binance_spot_testnet_multi
        asyncio.run(run_live_binance_spot_testnet_multi(
            syms,
            trade_qty=trade_qty,
            persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt,
            per_symbol_cap_usdt=per_symbol_cap_usdt,
            auto_close=auto_close,
            soft_cap_pct=soft_cap_pct,
            soft_cap_grace_sec=soft_cap_grace_sec
        ))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")


# --- comando multi-símbolo FUTURES TESTNET ---
@app.command()
def run_live_binance_futures_testnet_multi_cli(
    symbols: str = "BTC/USDT,ETH/USDT",
    leverage: int = 5,
    trade_qty: float = 0.001,
    persist_pg: bool = False,
    total_cap_usdt: float = 2000.0,
    per_symbol_cap_usdt: float = 1000.0,
    auto_close: bool = True,
    soft_cap_pct: float = 0.10,
    soft_cap_grace_sec: int = 30
):
    setup_logging()
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    import asyncio
    try:
        from ..live.runner_futures_testnet_multi import run_live_binance_futures_testnet_multi
        asyncio.run(run_live_binance_futures_testnet_multi(
            syms,
            leverage=leverage,
            trade_qty=trade_qty,
            persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt,
            per_symbol_cap_usdt=per_symbol_cap_usdt,
            auto_close=auto_close,
            soft_cap_pct=soft_cap_pct,
            soft_cap_grace_sec=soft_cap_grace_sec
        ))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")



def main():
    app()

if __name__ == "__main__":
    main()
