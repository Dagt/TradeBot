from types import SimpleNamespace

import pytest

from tradingbot.live._symbol_rules import denormalize_for_rules, resolve_symbol_rules


class _DummyMeta:
    def __init__(self):
        self.client = SimpleNamespace(symbols=[], markets={})

    def rules_for(self, symbol: str) -> SimpleNamespace:
        assert symbol == "ETH/USDT"
        return SimpleNamespace(
            qty_step=0.0001,
            price_step=0.01,
            min_notional=10,
            min_qty=0.0001,
        )


def test_symbol_rules_fallback_reconstructs_and_keeps_qty_step():
    meta = _DummyMeta()

    fetch_symbol = denormalize_for_rules("ETHUSDT", "ETHUSDT", meta)
    assert fetch_symbol == "ETH/USDT"

    rules, resolved = resolve_symbol_rules(meta, "ETHUSDT", "ETHUSDT")
    assert resolved == "ETH/USDT"
    assert rules.qty_step == pytest.approx(0.0001)
