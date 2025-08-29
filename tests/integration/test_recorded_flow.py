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
        initial_equity=df["close"].iloc[0],
        exchange_configs={"default": {"market_type": "spot"}},
    )
    risk = engine.risk[("alwaysbuy", sym)]
    engine.verbose_fills = True
    result = engine.run()
    assert result["fill_count"] == 1
    fills = pd.DataFrame(
        result["fills"],
        columns=[
            "timestamp",
            "bar_index",
            "order_id",
            "trade_id",
            "roundtrip_id",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_type",
            "fee_cost",
            "slip_bps",
            "slippage_pnl",
            "cash_after",
            "base_after",
            "equity_after",
            "realized_pnl",
            "realized_pnl_total",
        ],
    )
    assert len(result["fills"][0]) == 21
    assert (fills["cash_after"] >= -1e-9).all()
    assert (fills["base_after"] >= -1e-9).all()
    assert risk.rm.pos.qty == pytest.approx(0.0)
    assert risk.rm.pos.realized_pnl - fills["fee_cost"].sum() + fills["slippage_pnl"].sum() == pytest.approx(
        fills["realized_pnl"].sum()
    )
    final_price = df["close"].iloc[-1]
    expected_equity = fills["cash_after"].iloc[-1] + fills["base_after"].iloc[-1] * final_price
    assert result["equity"] == pytest.approx(expected_equity)
    assert result["equity"] == pytest.approx(103.206259139547)
    assert result["pnl"] == pytest.approx(2.706259139547001)
