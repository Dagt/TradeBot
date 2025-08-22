import pytest

from tradingbot.cli.main import get_adapter_class, get_supported_kinds


def test_get_supported_kinds_binance_spot():
    cls = get_adapter_class("binance_spot")
    assert cls is not None
    kinds = get_supported_kinds(cls)
    # basic kinds available on most venues
    assert "trades" in kinds
    assert "orderbook" in kinds
    # binance spot does not expose open interest
    assert "open_interest" not in kinds


def test_get_supported_kinds_binance_futures_ws():
    cls = get_adapter_class("binance_futures_ws")
    assert cls is not None
    kinds = get_supported_kinds(cls)
    assert "funding" in kinds
    assert "open_interest" in kinds


def test_get_supported_kinds_bybit_ws():
    cls = get_adapter_class("bybit_ws")
    assert cls is not None
    kinds = get_supported_kinds(cls)
    assert "funding" in kinds
    assert "open_interest" in kinds


def test_get_supported_kinds_okx_ws():
    cls = get_adapter_class("okx_ws")
    assert cls is not None
    kinds = get_supported_kinds(cls)
    assert "funding" in kinds
    assert "open_interest" in kinds
