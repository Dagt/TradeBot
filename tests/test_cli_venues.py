import pytest

from tradingbot.cli import main


def test_cli_venues_resolve_classes():
    for venue in main._AVAILABLE_VENUES:
        cls = main._get_adapter_cls(venue)
        assert getattr(cls, "name") == venue
