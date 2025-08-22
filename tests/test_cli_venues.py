import inspect

import pytest

from tradingbot.adapters.base import ExchangeAdapter
from tradingbot.cli.main import _AVAILABLE_VENUES, get_adapter_class


@pytest.mark.parametrize("venue", sorted(_AVAILABLE_VENUES))
def test_each_venue_loads_class(venue):
    cls = get_adapter_class(venue)
    assert cls is not None
    sig = inspect.signature(cls)
    kwargs = {}
    if "testnet" in sig.parameters:
        kwargs["testnet"] = True
    adapter = cls(**kwargs)
    assert isinstance(adapter, ExchangeAdapter)
