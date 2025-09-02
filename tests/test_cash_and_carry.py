import asyncio
import pytest

from tradingbot.bus import EventBus
from tradingbot.data.funding import poll_funding
from tradingbot.strategies.cash_and_carry import CashAndCarry, CashCarryConfig


class DummyAdapter:
    name = "dummy"

    def __init__(self):
        self.count = 0

    async def fetch_funding(self, symbol: str):
        self.count += 1
        return {"rate": 0.001 if self.count == 1 else -0.001, "interval_sec": 3600}


@pytest.mark.asyncio
async def test_poll_funding_publishes_events():
    adapter = DummyAdapter()
    bus = EventBus()
    received = []
    bus.subscribe("funding", lambda e: received.append(e))
    task = asyncio.create_task(poll_funding(adapter, "BTCUSDT", bus, interval=0.01))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert received and received[0]["rate"] == 0.001


def test_cash_and_carry_strategy():
    cfg = CashCarryConfig(symbol="BTCUSDT", threshold=0.0001)
    strat = CashAndCarry(cfg)
    sig_long = strat.on_bar({"spot": 100.0, "perp": 101.0, "funding": 0.001})
    assert sig_long and sig_long.side == "buy"
    sig_short = strat.on_bar({"spot": 100.0, "perp": 99.0, "funding": -0.001})
    assert sig_short and sig_short.side == "sell"
    sig_flat = strat.on_bar({"spot": 100.0, "perp": 100.0, "funding": 0.0001})
    assert sig_flat and sig_flat.side == "flat"


def test_cash_and_carry_intraday_threshold():
    cfg = CashCarryConfig(symbol="BTCUSDT", threshold=0.01)
    strat = CashAndCarry(cfg)
    bar = {"spot": 100.0, "perp": 101.0, "funding": 0.001, "timeframe": "1m"}
    sig = strat.on_bar(bar)
    assert sig and sig.side == "buy"
    sig_no_tf = strat.on_bar({"spot": 100.0, "perp": 101.0, "funding": 0.001})
    assert sig_no_tf and sig_no_tf.side == "flat"


def test_cash_and_carry_risk_closes_position():
    class DummyRisk:
        updated = False

        def get_trade(self, symbol):
            return {"symbol": symbol, "side": "buy"}

        def update_trailing(self, trade, price):
            self.updated = True

        def manage_position(self, trade, signal):
            return "close"

    cfg = CashCarryConfig(symbol="BTCUSDT", threshold=0.0001)
    rs = DummyRisk()
    strat = CashAndCarry(cfg, risk_service=rs)
    sig = strat.on_bar({"spot": 100.0, "perp": 101.0, "funding": 0.001, "symbol": "BTCUSDT"})
    assert rs.updated
    assert sig and sig.side == "sell" and sig.limit_price == 100.0
