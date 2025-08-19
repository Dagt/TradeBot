"""Common logging utilities.

This module centralises the creation of loggers so all components share the
same configuration and naming conventions.  Modules should obtain a logger via
``get_logger(__name__)`` instead of accessing :mod:`logging` directly.
"""

from __future__ import annotations

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger using the project's standard configuration.

    Parameters
    ----------
    name:
        Optional logger name.  When ``None`` the module name of this utility is
        used.
    """

    return logging.getLogger(name if name else __name__)


__all__ = ["get_logger"]

