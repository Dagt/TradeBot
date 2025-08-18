import pandas as pd
import pytest
from types import SimpleNamespace

from tradingbot.backtesting.engine import EventDrivenBacktestEngine, SlippageModel
from tradingbot.core import normalize
from tradingbot.strategies import STRATEGIES


class AlwaysBuyStrategy:
    name = "alwaysbuy"

    def on_bar(self, context):
        return SimpleNamespace(side="buy", strength=1.0)


@pytest.mark.integration
def test_recorded_full_flow_validates_fills_pnl_and_risk(monkeypatch):
    df = pd.read_csv("data/examples/btcusdt_1m.csv")
    monkeypatch.setitem(STRATEGIES, "alwaysbuy", AlwaysBuyStrategy)
    sym = normalize("BTC-USDT")
    engine = EventDrivenBacktestEngine(
        {sym: df},
        [("alwaysbuy", sym)],
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0),
    )
    risk = engine.risk[("alwaysbuy", sym)]
    risk.max_pos = 1.0
    result = engine.run()
    assert len(result["fills"]) == 1
    avg_price = result["orders"][0]["avg_price"]
    expected_pnl = df["close"].iloc[-1] - avg_price
    assert pytest.approx(result["equity"], rel=1e-4) == expected_pnl
    assert risk.pos.qty == 1.0
