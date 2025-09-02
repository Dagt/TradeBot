"""Utility helpers for the TradingBot command line interface."""

from __future__ import annotations

import ast
import inspect
import textwrap
import functools
from typing import Any

import typer

from .. import adapters
from ..adapters import (
    BinanceFuturesAdapter,
    BinanceSpotAdapter,
    BinanceFuturesWSAdapter,
    BinanceSpotWSAdapter,
    BybitFuturesAdapter,
    BybitSpotAdapter,
    BybitWSAdapter,
    DeribitAdapter,
    DeribitWSAdapter,
    OKXFuturesAdapter,
    OKXSpotAdapter,
    OKXWSAdapter,
)
from ..exchanges import SUPPORTED_EXCHANGES

# ---------------------------------------------------------------------------
# Typer option monkey patch
# ---------------------------------------------------------------------------
_original_option = getattr(typer.Option, "__wrapped__", typer.Option)


@functools.wraps(_original_option)
def _option(*args, **kwargs):
    if "type" in kwargs:
        kwargs["click_type"] = kwargs.pop("type")
    return _original_option(*args, **kwargs)


# Ensure the monkey patch is applied on import
typer.Option = _option  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------
# Manual mapping of venue names to adapter classes to avoid relying on
# capitalization conventions. This ensures acronyms such as OKX resolve
# correctly without deriving the class name dynamically.
_ADAPTER_CLASS_MAP: dict[str, type[adapters.ExchangeAdapter]] = {
    "binance_spot": BinanceSpotAdapter,
    "binance_futures": BinanceFuturesAdapter,
    "binance_spot_ws": BinanceSpotWSAdapter,
    "binance_futures_ws": BinanceFuturesWSAdapter,
    "bybit_spot": BybitSpotAdapter,
    "bybit_futures": BybitFuturesAdapter,
    "bybit_futures_ws": BybitWSAdapter,
    "okx_spot": OKXSpotAdapter,
    "okx_futures": OKXFuturesAdapter,
    "okx_futures_ws": OKXWSAdapter,
    "deribit_futures": DeribitAdapter,
    "deribit_futures_ws": DeribitWSAdapter,
}


def get_adapter_class(name: str) -> type[adapters.ExchangeAdapter] | None:
    """Return the adapter class for ``name`` if available."""

    return _ADAPTER_CLASS_MAP.get(name)


def get_supported_kinds(adapter_cls: type[adapters.ExchangeAdapter]) -> list[str]:
    """Return a sorted list of stream kinds supported by ``adapter_cls``."""

    kinds_attr = getattr(adapter_cls, "supported_kinds", None)
    if kinds_attr:
        return sorted(kinds_attr)

    kinds: set[str] = set()
    for name in dir(adapter_cls):
        if not name.startswith("stream_"):
            continue
        attr = getattr(adapter_cls, name)
        if not callable(attr):
            continue
        base_attr = getattr(adapters.ExchangeAdapter, name, None)
        if base_attr is attr:
            continue
        try:
            src = inspect.getsource(attr)
        except OSError:  # pragma: no cover - builtins or C extensions
            src = ""
        if src:
            fn_node = ast.parse(textwrap.dedent(src)).body[0]
            body = getattr(fn_node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            if len(body) == 1 and isinstance(body[0], ast.Raise):
                exc = body[0].exc
                if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                    continue
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id == "NotImplementedError"
                ):
                    continue
        kind = name[len("stream_"):]
        if kind in ("order_book", "orderbook"):
            kind = "orderbook"
        elif kind == "book_delta":
            kind = "delta"
        kinds.add(kind)
    name = getattr(adapter_cls, "name", "")
    if "futures" not in name:
        kinds.discard("funding")
        kinds.discard("open_interest")
    if not name.endswith("_ws"):
        kinds.add("bars")
    return sorted(kinds)


def _get_available_venues() -> set[str]:
    """Return venue names available for the CLI."""

    return set(_ADAPTER_CLASS_MAP)


_AVAILABLE_VENUES = _get_available_venues()
_WS_ONLY_VENUES = {v for v in SUPPORTED_EXCHANGES if v.endswith("_ws")}

_VENUE_CHOICES = ", ".join(sorted(SUPPORTED_EXCHANGES))


def _validate_venue(value: str) -> str:
    if value not in SUPPORTED_EXCHANGES:
        raise typer.BadParameter(
            f"Venue must be one of: {_VENUE_CHOICES}"
        )
    if value.endswith("_ws"):
        raise typer.BadParameter(
            "WebSocket venues are streaming-only and do not support trading or backfills"
        )
    return value


def _validate_backtest_venue(value: str) -> str:
    """Reject venues not suitable for backtesting."""

    if value.endswith("_testnet") or value.endswith("_ws"):
        raise typer.BadParameter(
            "Testnet and WebSocket venues are not supported; use *_spot or *_futures"
        )
    return _validate_venue(value)


def _parse_risk_pct(value: float) -> float:
    """Validate and normalise risk percentage values."""

    val = float(value)
    if val < 0:
        raise typer.BadParameter("risk-pct must be non-negative")
    if val >= 1:
        if val <= 100:
            return val / 100
        raise typer.BadParameter("risk-pct must be between 0 and 1")
    return val


def _parse_params(values: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` pairs into a dictionary."""

    params: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise typer.BadParameter("Parameters must be in key=value format")
        key, val = item.split("=", 1)
        try:
            params[key] = ast.literal_eval(val)
        except Exception:
            params[key] = val
    return params


__all__ = [
    "get_adapter_class",
    "get_supported_kinds",
    "_AVAILABLE_VENUES",
    "_WS_ONLY_VENUES",
    "_validate_venue",
    "_validate_backtest_venue",
    "_parse_risk_pct",
    "_parse_params",
]
