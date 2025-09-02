"""Command line entry point for TradingBot.

This module sets up common configuration and registers the different
command groups defined under :mod:`tradingbot.cli.commands`.
"""
from __future__ import annotations

import logging
import os
import sys

import typer

from tradingbot.utils.time_sync import check_ntp_offset

# Import utilities to ensure Typer patches and helpers are available
from . import utils as _utils  # noqa: F401
from .commands import backtesting, data, live

log = logging.getLogger(__name__)

_OFFSET_THRESHOLD = float(os.getenv("NTP_OFFSET_THRESHOLD", "1.0"))
if not os.getenv("TRADINGBOT_SKIP_NTP_CHECK"):
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

# Register subcommands
app.add_typer(data.app)
app.add_typer(backtesting.app)
app.add_typer(live.app)


def main() -> int:
    """Entry point used by ``python -m tradingbot.cli``."""
    try:
        app(standalone_mode=False)
        return 0
    except typer.Exit as exc:
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
