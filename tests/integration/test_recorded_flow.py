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
            "fee",
            "slip_bps",
            "cash_after",
            "base_after",
            "equity_after",
            "realized_pnl",
        ],
    )
    assert len(result["fills"][0]) == 19
    assert (fills["cash_after"] >= -1e-9).all()
    assert (fills["base_after"] >= -1e-9).all()
    assert risk.rm.pos.qty == pytest.approx(0.0)
    slip_cash_total = 0.0
    for row in fills.itertuples():
        slip_mult = row.slip_bps / 10000.0
        if row.side == "buy":
            place_price = row.price / (1 + slip_mult) if 1 + slip_mult != 0 else row.price
            slip_cash_total += (row.price - place_price) * row.qty
        else:
            place_price = row.price / (1 - slip_mult) if 1 - slip_mult != 0 else row.price
            slip_cash_total += (place_price - row.price) * row.qty
    assert risk.rm.pos.realized_pnl - fills["fee"].sum() - slip_cash_total == pytest.approx(
        fills["realized_pnl"].sum()
    )
    final_price = df["close"].iloc[-1]
    expected_equity = fills["cash_after"].iloc[-1] + fills["base_after"].iloc[-1] * final_price
    assert result["equity"] == pytest.approx(expected_equity)
    assert result["equity"] == pytest.approx(103.97112767590798)
    assert result["pnl"] == pytest.approx(3.4711276759079794)
