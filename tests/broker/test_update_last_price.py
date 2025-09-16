import pytest

from tradingbot.broker import Broker
from tradingbot.utils.metrics import FILL_COUNT, CANCELS


class DummyRisk:
    def __init__(self):
        self.calls = []

    def on_fill(self, symbol, side, qty, price=None, venue=None, atr=None):
        self.calls.append((symbol, side, qty, price, venue))


class DummyAdapter:
    name = "dummy"

    def __init__(self):
        self.risk_service = DummyRisk()
        self.pf_calls = 0
        self.exp_calls = 0

    def on_partial_fill(self, order, res):
        self.pf_calls += 1

    def on_order_expiry(self, order, res):
        self.exp_calls += 1

    def update_last_price(self, symbol, px):
        return [
            {
                "status": "partial",
                "symbol": symbol,
                "side": "buy",
                "qty": 1.0,
                "filled_qty": 1.0,
                "pending_qty": 0.5,
                "price": px,
            },
            {
                "status": "expired",
                "symbol": symbol,
                "side": "buy",
                "qty": 0.0,
                "filled_qty": 0.0,
                "pending_qty": 0.5,
                "price": px,
            },
        ]


def test_update_last_price_processes_fills():
    FILL_COUNT.clear()
    CANCELS._value.set(0)
    adapter = DummyAdapter()
    broker = Broker(adapter)

    broker.update_last_price("BTCUSDT", 100.0)

    samples = list(FILL_COUNT.collect())[0].samples
    fill_sample = [
        s
        for s in samples
        if s.name == "order_fills_total"
        and s.labels.get("symbol") == "BTCUSDT"
        and s.labels.get("side") == "buy"
    ][0]
    assert fill_sample.value == 1.0
    assert CANCELS._value.get() == 0.0
    assert adapter.pf_calls == 1
    assert adapter.exp_calls == 1
    assert adapter.risk_service.calls == [("BTCUSDT", "buy", 1.0, 100.0, "dummy")]
