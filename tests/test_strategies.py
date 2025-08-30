import pandas as pd
import yaml
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow
from tradingbot.strategies.mean_rev_ofi import MeanRevOFI
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.manager import RiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
import pytest
from hypothesis import given, strategies as st


def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0)

    sig_buy = strat.on_bar({"window": breakout_df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"
    sig_sell = strat.on_bar({"window": breakout_df_sell, "volatility": 0.0})
    assert sig_sell.side == "sell"


def test_breakout_atr_min_edge(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=200)
    assert strat.on_bar({"window": breakout_df_buy, "volatility": 0.0}) is None
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=100)
    assert (
        strat.on_bar({"window": breakout_df_buy, "volatility": 0.0}).side == "buy"
    )
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=1000)
    assert strat.on_bar({"window": breakout_df_sell, "volatility": 0.0}) is None
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, min_edge_bps=900)
    assert (
        strat.on_bar({"window": breakout_df_sell, "volatility": 0.0}).side
        == "sell"
    )


def test_breakout_atr_risk_service_handles_stop_and_size(breakout_df_buy):
    rm = RiskManager(risk_pct=0.02)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(rm, guard, risk_pct=0.02)
    svc.account.update_cash(1000.0)
    strat = BreakoutATR(ema_n=2, atr_n=2, mult=1.0, risk_service=svc)
    sig = strat.on_bar({"window": breakout_df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"])
    assert trade["stop"] == pytest.approx(expected_stop)


def test_order_flow_signals():
    df_buy = pd.DataFrame({
        "bid_qty": [1, 2, 3, 5],
        "ask_qty": [5, 4, 3, 2],
    })
    df_sell = pd.DataFrame({
        "bid_qty": [5, 4, 3, 2],
        "ask_qty": [1, 2, 3, 5],
    })
    strat = OrderFlow(window=3, buy_threshold=1.0, sell_threshold=1.0)
    sig_buy = strat.on_bar({"window": df_buy})
    assert sig_buy.side == "buy"
    sig_sell = strat.on_bar({"window": df_sell})
    assert sig_sell.side == "sell"


def test_mean_rev_ofi_signals():
    cfg = yaml.safe_load(
        """
ofi_window: 2
zscore_threshold: 0.5
vol_window: 2
vol_threshold: 1.0
"""
    )
    df_sell = pd.DataFrame(
        {
            "bid_qty": [1, 2, 10],
            "ask_qty": [3, 1, 0],
            "close": [100, 100, 100],
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0})
    assert sig_sell.side == "sell"

    df_buy = pd.DataFrame(
        {
            "bid_qty": [3, 1, 0],
            "ask_qty": [1, 2, 10],
            "close": [100, 100, 100],
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"


def test_mean_rev_ofi_trailing_stop_uses_atr():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {
        "side": "sell",
        "entry_price": 100.0,
        "qty": 1.0,
        "stop": 101.0,
        "atr": 1.0,
        "stage": 3,
    }
    rm.update_trailing(trade, 90.0)
    assert trade["stop"] == pytest.approx(90.0 + 2 * trade["atr"])


def test_breakout_vol_min_edge():
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=1100)
    assert strat.on_bar({"window": df_buy, "volatility": 0.0}) is None
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=1000)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig_buy.side == "buy"
    df_sell = pd.DataFrame({"close": [1, 2, 3, -5]})
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=2400)
    assert strat.on_bar({"window": df_sell, "volatility": 0.0}) is None
    strat = BreakoutVol(lookback=2, mult=0.5, min_edge_bps=2000)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0})
    assert sig_sell.side == "sell"


def test_breakout_vol_risk_service_handles_stop_and_size():
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
    rm = RiskManager(risk_pct=0.02)
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(rm, guard, risk_pct=0.02)
    svc.account.update_cash(1000.0)
    strat = BreakoutVol(lookback=2, mult=0.5, risk_service=svc)
    sig = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade.get("atr"))
    assert trade["stop"] == pytest.approx(expected_stop)


@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_buy_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start + i * inc for i in range(3)],
        "ask_qty": [start - i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig.side == "buy"


@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_sell_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start - i * inc for i in range(3)],
        "ask_qty": [start + i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig.side == "sell"
