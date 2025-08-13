# src/tradingbot/cli/main.py
import typer, logging, time
import pandas as pd

from ..logging_conf import setup_logging
from ..backtest.event_engine import run_backtest_csv
from ..strategies.breakout_atr import BreakoutATR
from ..risk.manager import RiskManager
from ..execution.paper import PaperAdapter
from ..live.runner import run_live_binance  # <-- NUEVO

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

def main():
    app()

if __name__ == "__main__":
    main()
