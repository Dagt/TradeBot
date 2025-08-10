import typer, logging
from ..logging_conf import setup_logging
from ..backtest.event_engine import run_backtest_csv

app = typer.Typer(add_completion=False)

@app.command()
def backtest(data: str, symbol: str = "BTC/USDT"):
    """Backtest simple desde un CSV con columnas: timestamp, open, high, low, close, volume"""
    setup_logging()
    res = run_backtest_csv(data, symbol=symbol)
    typer.echo(res)

def main():
    app()

if __name__ == "__main__":
    main()
