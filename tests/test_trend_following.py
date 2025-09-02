import pandas as pd
import pytest

from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.trend_following import TrendFollowing


def test_trend_following_trailing_stop_uses_atr():
    account = Account(float("inf"), cash=1000.0)
    rm = CoreRiskManager(account)
    trade = {
        "side": "buy",
        "entry_price": 100.0,
        "qty": 1.0,
        "stop": 99.0,
        "atr": 1.0,
        "stage": 3,
    }
    rm.update_trailing(trade, 110.0)
    assert trade["stop"] == pytest.approx(110.0 - 2 * trade["atr"])


def test_trend_following_risk_service_handles_stop_and_size():
    df = pd.DataFrame({"close": [1, 2, 3]})
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
    strat = TrendFollowing(rsi_n=2, **{"risk_service": svc})
    sig = strat.on_bar({"window": df, "atr": 1.0, "volatility": 0.0})
    assert sig and sig.side == "buy"
    assert sig.limit_price == pytest.approx(df["close"].iloc[-1])
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"])
    assert trade["stop"] == pytest.approx(expected_stop)


@pytest.mark.parametrize("timeframe", ["1m", "15m"])
def test_trend_following_generates_signal_across_timeframes(timeframe):
    lookback_min = 60
    tf_minutes = 1 if timeframe == "1m" else 15
    lookback_bars = lookback_min // tf_minutes
    n = max(lookback_bars * 2 + 1, 15)
    prices = [100] * (n - 1) + [110]
    freq = f"{tf_minutes}min"
    df = pd.DataFrame({"close": prices}, index=pd.date_range("2024", periods=n, freq=freq))
    strat = TrendFollowing(vol_lookback=lookback_min)
    bar = {"window": df, "timeframe": timeframe, "volatility": 0.0}
    sig = strat.on_bar(bar)
    assert sig and sig.side == "buy"

