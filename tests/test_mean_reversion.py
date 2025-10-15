import math

import pandas as pd
from tradingbot.execution.order_types import Order
from tradingbot.strategies import mean_reversion as mr
from tradingbot.strategies.base import Signal
from tradingbot.strategies.mean_reversion import (
    MeanReversion,
    _normalized_strength,
    generate_signals,
)

def test_mean_reversion_on_bar_signals():
    df_down = pd.DataFrame({"close": list(range(20, 0, -1))})
    df_up = pd.DataFrame({"close": list(range(1, 21))})
    strat = MeanReversion(rsi_n=5)
    sig_buy = strat.on_bar({"window": df_down, "volatility": 0.0})
    sig_sell = strat.on_bar({"window": df_up, "volatility": 0.0})
    assert sig_buy.side == "buy"
    assert sig_sell.side == "sell"


class ClampRiskService:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.min_order_qty = 0.0
        self.min_notional = 0.0
        self.pos = type("Pos", (), {"realized_pnl": 0.0, "qty": 0.0})()

    def calc_position_size(self, strength, price, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return float(strength)

    @staticmethod
    def initial_stop(price, side, atr):  # noqa: ANN001
        atr = float(atr or 0.0)
        if str(side).lower() == "buy":
            return float(price) - atr
        return float(price) + atr

    @staticmethod
    def get_trade(symbol):  # noqa: ANN001
        return None

    @staticmethod
    def update_trailing(trade, price):  # noqa: ANN001
        return None

    @staticmethod
    def manage_position(trade, sig):  # noqa: ANN001
        return "hold"

    @staticmethod
    def update_signal_strength(symbol, strength):  # noqa: ANN001
        return None


def test_mean_reversion_uses_maker_limits_and_clamp():
    prices = [12.0, 11.5, 11.0, 10.5, 10.0, 9.5, 9.0]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.2 for p in prices],
            "low": [p - 0.2 for p in prices],
            "close": prices,
            "volume": [100.0] * len(prices),
        }
    )
    best_bid = prices[-1] - 0.1
    best_ask = prices[-1] + 0.1
    risk = ClampRiskService()
    strat = MeanReversion(rsi_n=5, risk_service=risk)
    bar = {
        "window": df,
        "volatility": 0.0,
        "symbol": "X",
        "bid": best_bid,
        "ask": best_ask,
    }
    sig = strat.on_bar(bar)
    assert sig is not None and sig.side == "buy"
    assert sig.post_only is True
    assert 0.0 < sig.strength <= 1.0
    assert risk.calls and risk.calls[0].get("clamp") is False

    meta = sig.metadata
    anchor = meta["base_price"] + meta["limit_offset"]
    assert math.isclose(anchor, best_bid, rel_tol=1e-9, abs_tol=1e-9)
    assert sig.limit_price <= anchor + 1e-9

    order = Order("X", sig.side, "limit", 1.0, price=sig.limit_price, post_only=sig.post_only)
    res = {"price": best_bid, "pending_qty": order.qty}
    for _ in range(3):
        action = strat.on_order_expiry(order, res)
        assert action == "re_quote"
        assert order.price <= anchor + 1e-9

    assert math.isclose(strat.trade["qty"], sig.strength, rel_tol=1e-9, abs_tol=1e-9)

def test_mean_reversion_generate_signals():
    df = pd.DataFrame({"price": [1, 2, 3, 4, 3, 2, 1, 2, 3]})
    params = {"window": 3, "threshold": 0.5, "position_size": 1}
    res = generate_signals(df, params)
    assert {"signal", "position", "fee", "slippage"} <= set(res.columns)
    assert len(res) == len(df)


def _const_rsi(val: float):
    def fn(df: pd.DataFrame, n: int):  # noqa: ANN001
        return pd.Series([val] * len(df), index=df.index)
    return fn


def test_trend_detection_1m(monkeypatch):
    df = pd.DataFrame({"close": list(range(1, 100))})
    monkeypatch.setattr(mr, "rsi", _const_rsi(56))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (55, 45)
    )
    strat = MeanReversion(timeframe="1m")
    sig = strat.on_bar({"window": df})
    assert sig is None


def test_trend_detection_5m(monkeypatch):
    df = pd.DataFrame({"close": list(range(1, 100))})
    monkeypatch.setattr(mr, "rsi", _const_rsi(56))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (55, 45)
    )
    strat = MeanReversion(timeframe="5m")
    sig = strat.on_bar({"window": df})
    assert sig is None


def _strong_trend_df() -> pd.DataFrame:
    prices = []
    price = 100.0
    for idx in range(80):
        price *= 1.035 + (0.02 if idx % 2 == 0 else -0.01)
        prices.append(price)
    return pd.DataFrame({"close": prices})


def _slow_market_df() -> pd.DataFrame:
    prices = [100.0 + math.sin(i / 5.0) * 0.2 for i in range(80)]
    return pd.DataFrame({"close": prices})


def test_trend_filter_blocks_countertrend_sells(monkeypatch):
    df = _strong_trend_df()
    monkeypatch.setattr(mr, "rsi", _const_rsi(70))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (60, 40)
    )
    strat = MeanReversion(
        timeframe="1m",
        trend_ma_bps=40.0,
        trend_rsi_shift=6.0,
        min_volatility=0,
    )
    sig = strat.on_bar({"window": df})
    assert sig is None


def test_trend_filter_preserves_slow_rebounds(monkeypatch):
    df = _slow_market_df()
    monkeypatch.setattr(mr, "rsi", _const_rsi(70))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (60, 40)
    )
    strat = MeanReversion(
        timeframe="1m",
        trend_ma_bps=40.0,
        trend_rsi_shift=6.0,
        min_volatility=0,
    )
    sig = strat.on_bar({"window": df})
    assert sig is not None
    assert sig.side == "sell"


class DummyRiskService:
    def __init__(self, side: str = "buy") -> None:
        self._trade = {"side": side}
        self.min_order_qty = 0.0
        self.min_notional = 0.0

    def get_trade(self, symbol: str) -> dict | None:
        return self._trade

    def update_trailing(self, trade, price):
        return None

    def manage_position(self, trade, sig):
        return "hold"

    def calc_position_size(self, strength, price, **kwargs):  # noqa: ANN001
        return float(strength or 0.0)

    @staticmethod
    def initial_stop(price, side, atr):  # noqa: ANN001
        if str(side).lower() == "buy":
            return float(price) - float(atr or 0.0)
        return float(price) + float(atr or 0.0)

    @staticmethod
    def update_signal_strength(symbol, strength):  # noqa: ANN001
        return None


def test_mean_reversion_trend_exit_respects_min_hold():
    strat = MeanReversion(timeframe="1h", time_stop=9, min_volatility=0)
    trade = {"side": "buy"}
    bar = {"symbol": "X"}
    price = 100.0

    signal = strat._maybe_exit_for_trend(
        bar,
        price,
        trade,
        hold_bars=2,
        min_hold_bars=3,
        last_rsi=82.0,
        trend_dir=-1,
        market_state="trend",
    )
    assert signal is None

    signal = strat._maybe_exit_for_trend(
        bar,
        price,
        trade,
        hold_bars=3,
        min_hold_bars=3,
        last_rsi=82.0,
        trend_dir=-1,
        market_state="trend",
    )
    assert signal is not None
    assert signal.side == "sell"


def test_low_volatility_windows_block_signals(monkeypatch):
    monkeypatch.setattr(mr, "rsi", _const_rsi(20))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (60, 40)
    )

    prices = []
    price = 100.0
    for idx in range(60):
        price += 2.0 if idx % 2 == 0 else -2.0
        prices.append(price)
    for _ in range(20):
        price += 0.02
        prices.append(price)

    windows = [pd.DataFrame({"close": prices[: idx + 1]}) for idx in range(len(prices))]

    strat_filtered = MeanReversion(timeframe="1m", min_volatility=0.0)
    strat_baseline = MeanReversion(
        timeframe="1m",
        min_volatility=0.0,
        vol_floor_quantile=0.0,
    )

    filtered = [
        strat_filtered.on_bar({"window": window, "symbol": "LOW"})
        for window in windows
    ]
    baseline = [
        strat_baseline.on_bar({"window": window, "symbol": "LOW"})
        for window in windows
    ]

    assert any(sig is not None for sig in filtered[:-15])
    assert all(sig is None for sig in filtered[-5:])
    assert any(sig is not None for sig in baseline[-5:])


def test_mean_reversion_backtest_vol_floor_reduces_fees(monkeypatch):
    monkeypatch.setattr(mr, "rsi", _const_rsi(20))
    monkeypatch.setattr(
        MeanReversion, "auto_threshold", lambda self, series, **_: (60, 40)
    )

    prices = []
    segments = []
    price = 100.0
    for idx in range(160):
        price += 1.6 if idx % 2 == 0 else -1.4
        prices.append(price)
        segments.append("high")
    flat_price = price
    for _ in range(20):
        price = flat_price
        prices.append(price)
        segments.append("low")

    strat_baseline = MeanReversion(
        timeframe="3m",
        min_volatility=0.0,
        vol_floor_quantile=0.0,
    )
    strat_filtered = MeanReversion(
        timeframe="3m",
        min_volatility=0.0,
    )

    baseline_fee = 0.0
    filtered_fee = 0.0
    baseline_low_signals = 0
    filtered_low_signals = 0
    fee_per_trade = 0.0005

    for idx in range(len(prices)):
        window = pd.DataFrame({"close": prices[: idx + 1]})
        bar = {"window": window, "symbol": "SIM", "timeframe": "3m"}
        sig_base = strat_baseline.on_bar(bar)
        sig_filt = strat_filtered.on_bar(bar)
        if sig_base is not None:
            baseline_fee += abs(sig_base.strength) * fee_per_trade
            if segments[idx] == "low":
                baseline_low_signals += 1
        if sig_filt is not None:
            filtered_fee += abs(sig_filt.strength) * fee_per_trade
            if segments[idx] == "low":
                filtered_low_signals += 1

    assert baseline_low_signals > 0
    assert filtered_low_signals == 0
    assert filtered_fee < baseline_fee


def test_mean_reversion_relaxed_confirmation_backtest(monkeypatch):
    prices: list[float] = []
    rsi_values: list[float] = []
    base = 100.0
    for _ in range(4):
        p0 = base + 0.6
        p1 = base + 1.1
        p2 = base + 1.6
        p3 = p2 * (1 + 0.0012)
        p4 = p3 - 0.9
        p5 = p4 - 0.5
        p6 = p5 - 0.3
        prices.extend([p0, p1, p2, p3, p4, p5, p6])
        rsi_values.extend([55.0, 62.0, 68.5, 73.0, 69.5, 61.0, 45.0])
        base = p6 + 1.0

    df = pd.DataFrame({"close": prices})

    def fake_rsi(data: pd.DataFrame, n: int):  # noqa: ANN001
        return pd.Series(rsi_values[: len(data)], index=data.index)

    monkeypatch.setattr(mr, "rsi", fake_rsi)
    monkeypatch.setattr(
        MeanReversion,
        "auto_threshold",
        lambda self, series, **_: (60.0, 40.0),
    )

    strat = MeanReversion(timeframe="15m", rsi_n=5, min_volatility=0.0)
    closes = df["close"].to_numpy()

    fee_per_trade = 0.0004
    signals: list[tuple[int, Signal]] = []
    pnl_total = 0.0
    fee_total = 0.0

    for idx in range(len(df)):
        window = df.iloc[: idx + 1]
        sig = strat.on_bar({"window": window, "timeframe": "15m"})
        if sig is None:
            continue
        signals.append((idx, sig))
        strength = float(sig.strength)
        fee_total += abs(strength) * fee_per_trade
        exit_idx = min(idx + 2, len(df) - 1)
        entry_price = closes[idx]
        exit_price = closes[exit_idx]
        direction = 1.0 if sig.side == "buy" else -1.0
        pnl_total += direction * (exit_price - entry_price) / entry_price * strength

    assert signals, "Expected at least one fill under relaxed confirmation"

    upper = 60.0
    strict_indices: list[int] = []
    strict_fee = 0.0
    strict_pnl = 0.0
    for idx in range(1, len(df)):
        if idx < strat.rsi_n:
            continue
        last_rsi = rsi_values[idx]
        prev_rsi = rsi_values[idx - 1]
        if last_rsi <= upper:
            continue
        if prev_rsi > last_rsi and closes[idx] < closes[idx - 1]:
            strict_indices.append(idx)
            deviation = (last_rsi - upper) / max(1.0, 100.0 - upper)
            raw = max(0.0, deviation * 3.0)
            strength = _normalized_strength(raw)
            strict_fee += strength * fee_per_trade
            exit_idx = min(idx + 2, len(df) - 1)
            entry_price = closes[idx]
            exit_price = closes[exit_idx]
            strict_pnl += (-1.0) * (exit_price - entry_price) / entry_price * strength

    signalled_indices = {idx for idx, _ in signals}
    assert set(strict_indices) <= signalled_indices

    additional_indices = [idx for idx in signalled_indices if idx not in strict_indices]
    assert additional_indices, "Relaxed confirmation should add extra fills"

    assert pnl_total >= strict_pnl - 1e-6
    assert pnl_total > 0
    assert strict_indices, "Baseline strict confirmation should exist"
    assert strict_fee > 0
    assert fee_total <= strict_fee * 2.5
