"""Command line interface for TradeBot."""

import typer

from .config import Settings

app = typer.Typer(help="TradeBot command line utilities")


@app.command()
def run() -> None:
    """Run the trading bot (placeholder)."""
    settings = Settings()
    typer.echo(f"Starting TradeBot with mode={settings.mode}")


if __name__ == "__main__":  # pragma: no cover
    app()
