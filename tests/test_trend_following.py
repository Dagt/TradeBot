import math

import pandas as pd
import pytest

import tradingbot.strategies.trend_following as trend_following
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
    price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    tf_minutes = TrendFollowing._tf_minutes(strat.timeframe, strat.timeframe)
    lookback_bars = max(1, math.ceil(strat.vol_lookback / tf_minutes))
    ret = df["close"].pct_change().dropna()
    vol_series = ret.rolling(lookback_bars).std().dropna()
    vol_bps = float(vol_series.iloc[-1]) * 10000 if len(vol_series) else 0.0
    price_abs = price if price > 0 else 1.0
    min_offset = price_abs * 0.001
    vol_offset = price_abs * abs(vol_bps) / 10000.0 if vol_bps else 0.0
    entry_vol = max(min_offset, vol_offset)
    expected_limit = max(price + entry_vol, prev_close + entry_vol)
    assert sig.limit_price == pytest.approx(expected_limit)
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
    lookback_bars = math.ceil(lookback_min / tf_minutes)
    n = max(lookback_bars * 2 + 1, 15)
    prices = [100] * (n - 1) + [110]
    freq = f"{tf_minutes}min"
    df = pd.DataFrame({"close": prices}, index=pd.date_range("2024", periods=n, freq=freq))
    strat = TrendFollowing(vol_lookback=lookback_min)
    bar = {"window": df, "timeframe": timeframe, "volatility": 0.0}
    sig = strat.on_bar(bar)
    assert sig and sig.side == "buy"


def test_trend_following_volatility_lookback_covers_minutes():
    lookback_min = 50
    timeframe = "15m"
    tf_minutes = 15
    expected_bars = math.ceil(lookback_min / tf_minutes)
    prices = [100.0] * (expected_bars + 10)
    index = pd.date_range("2024", periods=len(prices), freq=f"{tf_minutes}min")
    df = pd.DataFrame({"close": prices}, index=index)

    class DummyRQ:
        def __init__(self):
            self.calls: list[tuple[int, int | None]] = []

        def get(self, symbol, name, *, window, q, min_periods=None):
            self.calls.append((window, min_periods))

            class _Tracker:
                def update(self_inner, value):
                    return value

            return _Tracker()

    strat = TrendFollowing(rsi_n=2, vol_lookback=lookback_min)
    tracker = DummyRQ()
    strat._rq = tracker
    bar = {"window": df, "timeframe": timeframe, "volatility": 0.0}
    strat.on_bar(bar)

    vol_calls = [call for call in tracker.calls if call[0] == expected_bars * 5]
    assert vol_calls, "Rolling quantile cache not queried with expected window"
    _, min_periods = vol_calls[0]
    assert min_periods == expected_bars


def test_trend_following_respects_explicit_min_volatility():
    prices = [100 + i * 0.01 for i in range(30)]
    df = pd.DataFrame({"close": prices})
    bar = {"window": df, "timeframe": "1m"}

    strat_override = TrendFollowing(min_volatility=5.0)
    sig_override = strat_override.on_bar(bar)
    assert sig_override is None
    assert strat_override.min_volatility == pytest.approx(5.0)

    strat_auto = TrendFollowing()
    sig_auto = strat_auto.on_bar(bar)
    assert sig_auto and sig_auto.side == "buy"


class DummyRiskService:
    def __init__(self, multiplier: float = 10.0):
        self.multiplier = multiplier
        self.calls: list[tuple[float, float]] = []
        self.min_order_qty = 0.0
        self.min_notional = 0.0

    def calc_position_size(self, strength, price, **kwargs):
        self.calls.append((strength, price))
        return float(strength) * self.multiplier

    @staticmethod
    def initial_stop(price, side, atr):
        atr = float(atr or 0.0)
        if side == "buy":
            return float(price) - atr
        return float(price) + atr


def _constant_series(length: int, value: float) -> pd.Series:
    return pd.Series([value] * length)


def _constant_ofi(length: int, value: float) -> pd.Series:
    data = [0.0] * (max(length, 1) - 1)
    data.append(value)
    return pd.Series(data)


def test_trend_following_strength_scales_with_rsi_distance_buy(monkeypatch):
    threshold = 70.0
    monkeypatch.setattr(
        TrendFollowing,
        "auto_threshold",
        lambda self, symbol, last_rsi, vol_bps: threshold,
    )
    monkeypatch.setattr(
        trend_following,
        "calc_ofi",
        lambda data: _constant_ofi(len(data), 1.0),
    )
    df = pd.DataFrame(
        {
            "close": [100.0] * 6,
            "bid_qty": [1.0] * 6,
            "ask_qty": [1.0] * 6,
        }
    )

    def run_case(rsi_value: float):
        monkeypatch.setattr(
            trend_following,
            "rsi",
            lambda data, n, value=rsi_value: _constant_series(len(data), value),
        )
        risk = DummyRiskService()
        strat = TrendFollowing(rsi_n=2, risk_service=risk)
        bar = {"window": df, "atr": 1.0, "volatility": 0.0}
        sig = strat.on_bar(bar)
        assert sig and sig.side == "buy"
        assert all(pytest.approx(sig.strength) == call[0] for call in risk.calls)
        assert strat.trade["qty"] == pytest.approx(sig.strength * risk.multiplier)
        return sig.strength, strat.trade["qty"]

    near_strength, near_qty = run_case(threshold + 0.5)
    far_strength, far_qty = run_case(99.0)

    assert 0 < near_strength < far_strength <= 1.0
    assert near_qty < far_qty


def test_trend_following_strength_scales_with_rsi_distance_sell(monkeypatch):
    threshold = 70.0
    monkeypatch.setattr(
        TrendFollowing,
        "auto_threshold",
        lambda self, symbol, last_rsi, vol_bps: threshold,
    )
    monkeypatch.setattr(
        trend_following,
        "calc_ofi",
        lambda data: _constant_ofi(len(data), -1.0),
    )
    df = pd.DataFrame(
        {
            "close": [100.0] * 6,
            "bid_qty": [1.0] * 6,
            "ask_qty": [1.0] * 6,
        }
    )

    def run_case(rsi_value: float):
        monkeypatch.setattr(
            trend_following,
            "rsi",
            lambda data, n, value=rsi_value: _constant_series(len(data), value),
        )
        risk = DummyRiskService()
        strat = TrendFollowing(rsi_n=2, risk_service=risk)
        bar = {"window": df, "atr": 1.0, "volatility": 0.0}
        sig = strat.on_bar(bar)
        assert sig and sig.side == "sell"
        assert all(pytest.approx(sig.strength) == call[0] for call in risk.calls)
        assert strat.trade["qty"] == pytest.approx(sig.strength * risk.multiplier)
        return sig.strength, strat.trade["qty"]

    near_strength, near_qty = run_case(100.0 - threshold - 0.5)
    far_strength, far_qty = run_case(1.0)

    assert 0 < near_strength < far_strength <= 1.0
    assert near_qty < far_qty

