"""Subcommands for the TradingBot CLI."""

from __future__ import annotations

import typer

# Each module exposes its own ``app`` Typer instance.  They are imported in
# :mod:`tradingbot.cli.main` and registered with ``app.add_typer`` so their
# commands are available at the top level CLI.
