import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


class AlwaysBuyStrategy:
    name = "alwaysbuy"

    def on_bar(self, context):
        return SimpleNamespace(side="buy", strength=1.0)


@pytest.mark.integration
def test_inferred_slippage_applies_spread_and_participation(monkeypatch):
    symbol = "BTCUSDT"
    idx = pd.date_range("2024-01-01", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000.0] * 5,
        },
        index=idx,
    )
    monkeypatch.setitem(STRATEGIES, "alwaysbuy", AlwaysBuyStrategy)
    engine = EventDrivenBacktestEngine(
        {symbol: df},
        [("alwaysbuy", symbol)],
        latency=1,
        window=1,
        slippage=None,
        exchange_configs={"default": {"market_type": "spot"}},
    )
    result = engine.run()
    assert result["orders"]
    model = engine.slippage
    assert model.max_bar_participation is not None
    assert 0.0 < model.max_bar_participation <= 0.2
    assert model.base_spread > 0.0
    qty = 5000.0
    bar = {"volume": 1000.0, "close": 100.0}
    price = 100.0
    adj_price, fill_qty, _ = model.fill("buy", qty, price, bar)
    expected_cap = bar["volume"] * model.max_bar_participation
    assert fill_qty == pytest.approx(expected_cap)
    assert adj_price > price
