import pandas as pd
import yaml
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow
from tradingbot.strategies.mean_rev_ofi import MeanRevOFI
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
import pytest
from hypothesis import given, strategies as st, settings
from tradingbot.data.features import keltner_channels, atr


def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2)

    sig_buy = strat.on_bar({"window": breakout_df_buy, "volatility": 0.0})
    atr_buy = atr(breakout_df_buy, 2).dropna().iloc[-1]
    upper, lower = keltner_channels(breakout_df_buy, 2, 2, strat.mult)
    assert sig_buy.side == "buy"
    assert sig_buy.limit_price == pytest.approx(upper.iloc[-1] + 0.1 * atr_buy)

    sig_sell = strat.on_bar({"window": breakout_df_sell, "volatility": 0.0})
    atr_sell = atr(breakout_df_sell, 2).dropna().iloc[-1]
    upper, lower = keltner_channels(breakout_df_sell, 2, 2, strat.mult)
    assert sig_sell.side == "sell"
    assert sig_sell.limit_price == pytest.approx(lower.iloc[-1] - 0.1 * atr_sell)


def test_breakout_atr_risk_service_handles_stop_and_size(breakout_df_buy):
    account = Account(float("inf"))
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutATR(ema_n=2, atr_n=2, **{"risk_service": svc})
    sig = strat.on_bar({"window": breakout_df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"])
    assert trade["stop"] == pytest.approx(expected_stop)


def test_order_flow_signal_1m():
    idx = pd.date_range("2024-01-01", periods=6, freq="1min")
    df_buy = pd.DataFrame(
        {
            "bid_qty": [10, 12, 15, 20, 25, 30],
            "ask_qty": [30, 25, 20, 18, 15, 12],
            "close": [100, 100.1, 99.9, 100.3, 100.0, 100.5],
        },
        index=idx,
    )
    strat = OrderFlow(window=3, buy_threshold=0.5, sell_threshold=0.5)
    sig_buy = strat.on_bar({"window": df_buy})
    assert sig_buy and sig_buy.side == "buy"


def test_order_flow_signal_15m():
    idx = pd.date_range("2024-01-01", periods=6, freq="15min")
    df_sell = pd.DataFrame(
        {
            "bid_qty": [30, 25, 20, 18, 15, 12],
            "ask_qty": [10, 12, 15, 20, 25, 30],
            "close": [100, 99.8, 100.2, 99.7, 100.1, 99.6],
        },
        index=idx,
    )
    strat = OrderFlow(window=3, buy_threshold=0.5, sell_threshold=0.5)
    sig_sell = strat.on_bar({"window": df_sell})
    assert sig_sell and sig_sell.side == "sell"


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


def test_breakout_vol_risk_service_handles_stop_and_size():
    df_buy = pd.DataFrame({"close": [1, 2, 3, 10]})
    account = Account(float("inf"))
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutVol(lookback=2, **{"risk_service": svc})
    sig = strat.on_bar({"window": df_buy, "volatility": 0.0})
    assert sig and sig.side == "buy"
    assert sig.limit_price == pytest.approx(df_buy["close"].iloc[-1])
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade.get("atr"))
    assert trade["stop"] == pytest.approx(expected_stop)


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_buy_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start + i * inc for i in range(3)],
        "ask_qty": [start - i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_sell_property(start, inc):
    df = pd.DataFrame({
        "bid_qty": [start - i * inc for i in range(3)],
        "ask_qty": [start + i * inc for i in range(3)],
    })
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None
