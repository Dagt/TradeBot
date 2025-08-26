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
        equity_pct=1.0,
        equity_actual=df["close"].iloc[0],
    )
    risk = engine.risk[("alwaysbuy", sym)]
    result = engine.run()
    assert len(result["fills"]) > 0
    avg_price = result["orders"][0]["avg_price"]
    qty = risk.pos.qty
    expected_pnl = qty * (df["close"].iloc[-1] - avg_price)
    assert abs(result["equity"] - expected_pnl) < 0.01
    assert qty > 0
