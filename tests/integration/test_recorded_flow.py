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
    df = pd.read_csv("data/examples/btcusdt_3m.csv")
    monkeypatch.setitem(STRATEGIES, "alwaysbuy", AlwaysBuyStrategy)
    sym = normalize("BTC-USDT")
    engine = EventDrivenBacktestEngine(
        {sym: df},
        [("alwaysbuy", sym)],
        latency=1,
        window=1,
        slippage=SlippageModel(volume_impact=0.0, pct=0.0),
        initial_equity=df["close"].iloc[0],
        exchange_configs={"default": {"market_type": "spot"}},
        slippage_bps=0.0,
    )
    risk = engine.risk[("alwaysbuy", sym)]
    engine.verbose_fills = True
    result = engine.run()
    assert result["fill_count"] == 2
    assert result["order_count"] == len(result["orders"])
    assert result["cancel_count"] == 1
    assert result["cancel_ratio"] == pytest.approx(1 / result["order_count"])
    fills = pd.DataFrame(
        result["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )
    assert len(result["fills"][0]) == 13
    cash = engine.initial_equity
    base = 0.0
    for row in fills.itertuples():
        if row.side == "buy":
            cash -= row.price * row.qty + row.fee_cost
            base += row.qty
        else:
            cash += row.price * row.qty - row.fee_cost
            base -= row.qty
        assert cash >= -1e-9
        assert base >= -1e-9
    assert risk.pos.qty == pytest.approx(0.0)
    assert risk.pos.realized_pnl == pytest.approx(fills["realized_pnl"].sum())
    final_price = df["close"].iloc[-1]
    expected_equity = cash + base * final_price
    assert result["equity"] == pytest.approx(expected_equity)
    assert result["equity"] == pytest.approx(104.07283406244801)
    assert result["pnl"] == pytest.approx(3.5728340624480097)
