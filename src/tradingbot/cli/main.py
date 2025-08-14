# src/tradingbot/cli/main.py
import typer, logging, time
import pandas as pd

from ..logging_conf import setup_logging
from ..backtest.event_engine import run_backtest_csv
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.paper import PaperAdapter
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
def backtest(data: str, symbol: str = "BTC/USDT"):
    """Backtest vectorizado simple desde CSV (columnas: timestamp, open, high, low, close, volume)"""
    setup_logging()
    res = run_backtest_csv(data, symbol=symbol)
    typer.echo(res)

@app.command()
def run_csv_paper(data: str, symbol: str = "BTC/USDT", sleep_ms: int = 50, max_bars: int = 0):
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

    strat = BreakoutATR()
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
    auto_close: bool = True
):
    setup_logging()
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    import asyncio
    try:
        asyncio.run(run_live_binance_spot_testnet_multi(
            syms, trade_qty=trade_qty, persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt, per_symbol_cap_usdt=per_symbol_cap_usdt,
            auto_close=auto_close
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
    auto_close: bool = True
):
    setup_logging()
    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    import asyncio
    try:
        asyncio.run(run_live_binance_futures_testnet_multi(
            syms, leverage=leverage, trade_qty=trade_qty, persist_pg=persist_pg,
            total_cap_usdt=total_cap_usdt, per_symbol_cap_usdt=per_symbol_cap_usdt,
            auto_close=auto_close
        ))
    except KeyboardInterrupt:
        print("Detenido por el usuario.")

def main():
    app()

if __name__ == "__main__":
    main()
