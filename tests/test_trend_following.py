import math

import pandas as pd
import pytest

import tradingbot.strategies.trend_following as trend_following
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.execution.order_types import Order
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.strategies.trend_following import TrendFollowing


def _expected_lookback(strat: TrendFollowing, tf_minutes: int) -> int:
    min_bars = 3 if strat.vol_lookback >= tf_minutes else 2
    return max(min_bars, math.ceil(strat.vol_lookback / tf_minutes))


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
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6]})
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
    price = df["close"].iloc[-1]
    best_bid = price - 0.05
    best_ask = price + 0.05
    bar = {
        "window": df,
        "atr": 1.0,
        "volatility": 0.0,
        "symbol": "X",
        "bid": best_bid,
        "ask": best_ask,
    }
    sig = strat.on_bar(bar)
    assert sig and sig.side == "buy"
    assert sig.post_only is True
    assert 0.0 < sig.strength <= 1.0
    price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    tf_minutes = TrendFollowing._tf_minutes(strat.timeframe, strat.timeframe)
    lookback_bars = _expected_lookback(strat, tf_minutes)
    ret = df["close"].pct_change().dropna()
    vol_series = ret.rolling(lookback_bars).std().dropna()
    vol_bps = float(vol_series.iloc[-1]) * 10000 if len(vol_series) else 0.0
    price_abs = price if price > 0 else 1.0
    min_offset = price_abs * 0.001
    vol_offset = price_abs * abs(vol_bps) / 10000.0 if vol_bps else 0.0
    entry_vol = max(min_offset, vol_offset)
    meta = sig.metadata
    anchor = meta["base_price"] + meta["limit_offset"]
    assert math.isclose(anchor, best_bid, rel_tol=1e-9, abs_tol=1e-9)
    assert sig.limit_price <= anchor + 1e-9
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"], clamp=False)
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"])
    assert trade["stop"] == pytest.approx(expected_stop)

    order = Order("X", sig.side, "limit", 1.0, price=sig.limit_price, post_only=sig.post_only)
    res = {"price": best_bid, "pending_qty": order.qty}
    for _ in range(3):
        action = strat.on_order_expiry(order, res)
        assert action == "re_quote"
        assert order.price <= anchor + 1e-9


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


def test_trend_following_positive_vol_bps_with_short_minute_window(monkeypatch):
    timeframe = "15m"
    strat = TrendFollowing(timeframe=timeframe, vol_lookback=5, rsi_n=6, min_volatility=0.0)
    tracker = TrackingRQ()
    strat._rq = tracker
    captured: list[float] = []

    original_auto = TrendFollowing.auto_threshold

    def _capture(self, symbol, last_rsi, vol_bps):
        captured.append(vol_bps)
        return original_auto(self, symbol, last_rsi, vol_bps)

    monkeypatch.setattr(TrendFollowing, "auto_threshold", _capture)

    freq = "15min"
    prices = [100.0 + 0.05 * i for i in range(40)]
    df = pd.DataFrame({"close": prices}, index=pd.date_range("2024-01-01", periods=len(prices), freq=freq))
    bar = {"window": df, "timeframe": timeframe, "symbol": "SYM", "volatility": 0.0}

    strat.on_bar(bar)

    assert captured, "auto_threshold did not capture any volatility"
    assert captured[-1] > 0

    tf_minutes = TrendFollowing._tf_minutes(timeframe, strat.timeframe)
    lookback = _expected_lookback(strat, tf_minutes)
    tracker_key = ("SYM", "vol_bps")
    assert tracker_key in tracker.trackers
    tracker_state = tracker.trackers[tracker_key]
    assert tracker_state.min_periods == lookback
    assert tracker_state.window >= lookback
    assert tracker_state.values, "volatility quantile tracker did not receive values"
    assert tracker_state.values[-1] == pytest.approx(captured[-1])


def test_trend_following_rsi_threshold_tracks_volatility_regimes(monkeypatch):
    timeframe = "15m"
    strat = TrendFollowing(timeframe=timeframe, vol_lookback=5, rsi_n=6, min_volatility=0.0)
    tracker = TrackingRQ()
    strat._rq = tracker

    thresholds: list[tuple[float, float]] = []
    original_auto = TrendFollowing.auto_threshold

    def _capture(self, symbol, last_rsi, vol_bps):
        thresh = original_auto(self, symbol, last_rsi, vol_bps)
        thresholds.append((vol_bps, thresh))
        return thresh

    monkeypatch.setattr(TrendFollowing, "auto_threshold", _capture)

    freq = "15min"
    base_price = 100.0
    low_vol = [base_price + 0.05 * i for i in range(50)]
    high_vol = []
    price = base_price
    for i in range(50):
        price += 1.5 if i % 2 == 0 else -1.2
        high_vol.append(price)

    df_low = pd.DataFrame({"close": low_vol}, index=pd.date_range("2024-01-01", periods=len(low_vol), freq=freq))
    df_high = pd.DataFrame({"close": high_vol}, index=pd.date_range("2024-03-01", periods=len(high_vol), freq=freq))

    bar_low = {"window": df_low, "timeframe": timeframe, "symbol": "SYM", "volatility": 0.0}
    bar_high = {"window": df_high, "timeframe": timeframe, "symbol": "SYM", "volatility": 0.0}

    strat.on_bar(bar_low)
    strat.on_bar(bar_high)

    assert len(thresholds) >= 2
    low_vol_bps, low_threshold = thresholds[0]
    high_vol_bps, high_threshold = thresholds[-1]

    assert low_vol_bps > 0
    assert high_vol_bps > low_vol_bps
    assert high_threshold > low_threshold

class DummyRiskService:
    def __init__(self, multiplier: float = 10.0):
        self.multiplier = multiplier
        self.calls: list[dict[str, float | bool | None]] = []
        self.min_order_qty = 0.0
        self.min_notional = 0.0

    def calc_position_size(self, strength, price, **kwargs):
        self.calls.append(
            {
                "strength": float(strength),
                "price": float(price),
                "clamp": kwargs.get("clamp"),
            }
        )
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


class TrackingRQ:
    class Tracker:
        def __init__(self, window: int, q: float, min_periods: int | None) -> None:
            self.window = int(window)
            self.q = float(q)
            self.min_periods = int(min_periods) if min_periods is not None else self.window
            self.values: list[float] = []

        def update(self, value: float) -> float:
            self.values.append(float(value))
            if len(self.values) > self.window:
                self.values = self.values[-self.window :]
            if len(self.values) < self.min_periods:
                return math.nan
            ordered = sorted(self.values[-self.window :])
            if not ordered:
                return math.nan
            k = (len(ordered) - 1) * self.q
            low = int(math.floor(k))
            high = min(len(ordered) - 1, low + 1)
            frac = k - low
            if high == low:
                return ordered[low]
            return ordered[low] * (1 - frac) + ordered[high] * frac

    def __init__(self) -> None:
        self.trackers: dict[tuple[str, str], TrackingRQ.Tracker] = {}

    def get(
        self,
        symbol: str,
        name: str,
        *,
        window: int,
        q: float,
        min_periods: int | None = None,
    ) -> "TrackingRQ.Tracker":
        key = (symbol, name)
        tracker = self.trackers.get(key)
        if (
            tracker is None
            or tracker.window != int(window)
            or tracker.q != float(q)
            or tracker.min_periods
            != (int(min_periods) if min_periods is not None else int(window))
        ):
            tracker = TrackingRQ.Tracker(window, q, min_periods)
            self.trackers[key] = tracker
        return tracker


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
        assert all(pytest.approx(sig.strength) == call["strength"] for call in risk.calls)
        assert risk.calls and risk.calls[0]["clamp"] is False
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
        assert all(pytest.approx(sig.strength) == call["strength"] for call in risk.calls)
        assert risk.calls and risk.calls[0]["clamp"] is False
        assert strat.trade["qty"] == pytest.approx(sig.strength * risk.multiplier)
        return sig.strength, strat.trade["qty"]

    near_strength, near_qty = run_case(100.0 - threshold - 0.5)
    far_strength, far_qty = run_case(1.0)

    assert 0 < near_strength < far_strength <= 1.0
    assert near_qty < far_qty

