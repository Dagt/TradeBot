from __future__ import annotations

import re
from typing import Any, Iterable

from ..core.symbols import normalize

_DEFAULT_QUOTES = [
    "USDT",
    "FDUSD",
    "TUSD",
    "USDC",
    "BUSD",
    "BTC",
    "ETH",
    "BNB",
]


def _iter_sources(meta: Any) -> Iterable[Any]:
    if meta is not None:
        yield meta
        client = getattr(meta, "client", None)
        if client is not None:
            yield client


def _collect_quotes(meta: Any) -> list[str]:
    quotes: set[str] = set()
    for source in _iter_sources(meta):
        markets = getattr(source, "markets", None)
        if isinstance(markets, dict):
            for details in markets.values():
                quote = details.get("quote") if isinstance(details, dict) else None
                if isinstance(quote, str) and quote:
                    quotes.add(quote.upper())
        extra_quotes = getattr(source, "quotes", None)
        if isinstance(extra_quotes, Iterable) and not isinstance(extra_quotes, (str, bytes)):
            for quote in extra_quotes:
                if isinstance(quote, str) and quote:
                    quotes.add(quote.upper())
    quotes.update(_DEFAULT_QUOTES)
    return sorted(quotes, key=len, reverse=True)


def denormalize_for_rules(
    raw_symbol: str | None,
    normalized_symbol: str,
    meta: Any | None = None,
) -> str | None:
    """Rebuild the venue formatted symbol used by ``rules_for``."""
    if raw_symbol:
        raw = raw_symbol.upper()
        if "/" in raw:
            return raw
    normalized = normalize(raw_symbol or normalized_symbol)
    base_part, _, suffix = normalized.partition("-")
    quotes = _collect_quotes(meta)
    for quote in quotes:
        if base_part.endswith(quote):
            base = base_part[: -len(quote)]
            if not base:
                continue
            symbol = f"{base}/{quote}"
            if suffix:
                symbol = f"{symbol}-{suffix}"
            return symbol
    match = re.match(r"([A-Z0-9]+?)([A-Z]{2,6})$", base_part)
    if match:
        base, quote = match.groups()
        symbol = f"{base}/{quote}"
        if suffix:
            symbol = f"{symbol}-{suffix}"
        return symbol
    if raw_symbol:
        candidate = raw_symbol.replace("-", "/").upper()
        if "/" in candidate:
            return candidate
    return None


def resolve_symbol_rules(
    meta: Any,
    raw_symbol: str | None,
    normalized_symbol: str,
):
    """Return venue ``rules`` and the resolved symbol identifier."""
    if meta is None:
        raise LookupError("symbol metadata unavailable")
    fetch_symbol: str | None = None
    client = getattr(meta, "client", None)
    symbols = getattr(client, "symbols", None)
    if isinstance(symbols, Iterable) and not isinstance(symbols, (str, bytes)):
        for symbol in symbols:
            if not isinstance(symbol, str):
                continue
            try:
                if normalize(symbol) == normalized_symbol:
                    fetch_symbol = symbol
                    break
            except Exception:
                continue
    if fetch_symbol is None:
        fetch_symbol = denormalize_for_rules(raw_symbol, normalized_symbol, meta)
    if not fetch_symbol:
        raise LookupError(f"unable to determine venue symbol for {normalized_symbol}")
    rules = meta.rules_for(fetch_symbol)
    return rules, fetch_symbol
