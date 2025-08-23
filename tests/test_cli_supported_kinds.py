import pytest

from tradingbot.cli.main import get_adapter_class, get_supported_kinds

EXPECTED_KINDS = {
    "binance_spot": {
        "trades",
        "orderbook",
        "bba",
        "delta",
    },
    "binance_futures": {
        "trades",
        "orderbook",
        "bba",
        "delta",
        "funding",
        "open_interest",
    },
    "binance_spot_ws": {"trades", "trades_multi", "orderbook", "bba", "delta"},
    "binance_futures_ws": {
        "trades",
        "trades_multi",
        "orderbook",
        "bba",
        "delta",
        "funding",
    },
    "bybit_spot": {
        "trades",
        "orderbook",
        "bba",
        "delta",
    },
    "bybit_futures": {
        "trades",
        "orderbook",
        "bba",
        "delta",
        "funding",
        "open_interest",
    },
    "bybit_futures_ws": {
        "trades",
        "orderbook",
        "bba",
        "delta",
        "funding",
        "open_interest",
    },
    "okx_spot": {
        "trades",
        "orderbook",
        "bba",
        "delta",
    },
    "okx_futures": {
        "trades",
        "orderbook",
        "bba",
        "delta",
        "funding",
        "open_interest",
    },
    "okx_futures_ws": {
        "trades",
        "orderbook",
        "bba",
        "delta",
        "funding",
        "open_interest",
    },
    "deribit_futures": {"trades", "funding", "open_interest"},
    "deribit_futures_ws": {
        "trades",
        "orderbook",
        "bba",
        "delta",
    },
}


@pytest.mark.parametrize("name,expected", EXPECTED_KINDS.items())
def test_get_supported_kinds(name: str, expected: set[str]) -> None:
    cls = get_adapter_class(name)
    assert cls is not None
    kinds = set(get_supported_kinds(cls))
    assert kinds == expected

