import pandas as pd
import pytest

from tradingbot.strategies.depth_imbalance import DepthImbalance
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.core.account import Account as CoreAccount
from unittest.mock import MagicMock


def test_depth_imbalance_strategy_buy():
    df = pd.DataFrame({"bid_qty": [5, 7, 9], "ask_qty": [5, 5, 3]})
    strat = DepthImbalance(window=2, percentile=80)
    sig = strat.on_bar({"window": df, "close": 100.0})
    assert sig is not None and sig.side == "buy"
    assert sig.limit_price == 100.0


def test_depth_imbalance_strategy_sell():
    df = pd.DataFrame({"bid_qty": [5, 4, 3], "ask_qty": [5, 6, 7]})
    strat = DepthImbalance(window=2, percentile=80)
    sig = strat.on_bar({"window": df, "close": 100.0})
    assert sig is not None and sig.side == "sell"
    assert sig.limit_price == 100.0


def test_depth_imbalance_trailing_close():
    df = pd.DataFrame({"bid_qty": [5, 7, 9], "ask_qty": [5, 5, 3]})
    guard = PortfolioGuard(GuardConfig(venue="test"))
    account = CoreAccount(float("inf"), cash=1000.0)
    svc = RiskService(guard, account=account, risk_per_trade=1.0, risk_pct=0.01)
    svc.update_trailing = MagicMock(wraps=svc.update_trailing)
    svc.manage_position = MagicMock(wraps=svc.manage_position)
    strat = DepthImbalance(window=2, percentile=80, risk_service=svc)

    sig = strat.on_bar({"window": df, "close": 100.0})
    assert sig is not None and sig.side == "buy"
    assert svc.update_trailing.call_count == 1
    stop = strat.trade["stop"]

    sig2 = strat.on_bar({"window": df, "close": stop - 1})
    assert svc.update_trailing.call_count == 2
    assert svc.manage_position.call_count == 1
    assert sig2 is not None and sig2.side == "sell"
    assert strat.trade is None


@pytest.mark.parametrize("freq", ["1min", "5min", "15min"], ids=["1m", "5m", "15m"])
def test_depth_imbalance_timeframes(freq):
    idx = pd.date_range("2023", periods=3, freq=freq)
    df_buy = pd.DataFrame({"bid_qty": [5, 7, 9], "ask_qty": [5, 5, 3]}, index=idx)
    strat_buy = DepthImbalance(window=2, percentile=80)
    sig_buy = strat_buy.on_bar({"window": df_buy, "close": 100.0})
    assert sig_buy is not None and sig_buy.side == "buy"

    df_sell = pd.DataFrame({"bid_qty": [5, 4, 3], "ask_qty": [5, 6, 7]}, index=idx)
    strat_sell = DepthImbalance(window=2, percentile=80)
    sig_sell = strat_sell.on_bar({"window": df_sell, "close": 100.0})
    assert sig_sell is not None and sig_sell.side == "sell"
