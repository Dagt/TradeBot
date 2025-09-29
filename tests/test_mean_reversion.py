import math

import pandas as pd
from tradingbot.execution.order_types import Order
from tradingbot.strategies import mean_reversion as mr
from tradingbot.strategies.base import timeframe_to_minutes
from tradingbot.strategies.mean_reversion import MeanReversion, generate_signals

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
    assert risk.calls and risk.calls[0].get("clamp") is True

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
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (55, 45))
    strat = MeanReversion(timeframe="1m")
    sig = strat.on_bar({"window": df})
    assert sig is None


def test_trend_detection_5m(monkeypatch):
    df = pd.DataFrame({"close": list(range(1, 100))})
    monkeypatch.setattr(mr, "rsi", _const_rsi(56))
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (55, 45))
    strat = MeanReversion(timeframe="5m")
    sig = strat.on_bar({"window": df})
    assert sig is None


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


def test_mean_reversion_multi_timeframe_time_stop(monkeypatch):
    df = pd.DataFrame(
        {
            "open": [100 + 0.2 * i for i in range(60)],
            "high": [100 + 0.2 * i + 0.1 for i in range(60)],
            "low": [100 + 0.2 * i - 0.1 for i in range(60)],
            "close": [100 + 0.2 * i for i in range(60)],
            "volume": [50.0] * 60,
        }
    )
    monkeypatch.setattr(mr, "rsi", _const_rsi(50))
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (60, 40))

    risk = DummyRiskService("buy")
    strat = MeanReversion(timeframe="1h", time_stop=6, min_volatility=0, risk_service=risk)
    bar_minutes = timeframe_to_minutes("4h")
    expected_bars = max(
        strat._min_time_stop_bars or 1,
        math.ceil(
            strat._time_stop_target_bars
            * strat._base_timeframe_minutes
            / bar_minutes
        ),
    )
    assert expected_bars >= 3

    symbol = "X"
    for idx in range(1, expected_bars):
        sig = strat.on_bar({"window": df, "timeframe": "4h", "symbol": symbol, "volume": 50.0})
        assert sig is None
        assert strat.time_stop == expected_bars
        assert strat._open_bars[symbol] == idx

    exit_sig = strat.on_bar({"window": df, "timeframe": "4h", "symbol": symbol, "volume": 50.0})
    assert exit_sig is not None
    assert exit_sig.side == "sell"
    assert strat.time_stop == expected_bars
    assert strat._open_bars[symbol] == expected_bars


def test_mean_reversion_volatility_floor_blocks_low_vol(monkeypatch):
    high_prices = [100, 112, 94, 118, 90, 122, 88, 126]
    low_prices = [100.0, 100.1, 99.95, 100.08, 100.02, 100.05, 100.01, 100.03]
    high_df = pd.DataFrame({"close": high_prices})
    low_df = pd.DataFrame({"close": low_prices})

    def fake_rsi(df: pd.DataFrame, n: int) -> pd.Series:  # noqa: ANN001
        pattern = [70.0 if i % 2 == 0 else 30.0 for i in range(len(df))]
        return pd.Series(pattern, index=df.index)

    monkeypatch.setattr(mr, "rsi", fake_rsi)
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (60, 40))

    symbol = "TEST"
    strat = MeanReversion(
        timeframe="3m",
        rsi_n=5,
        vol_floor_window=4,
        vol_floor_min_periods=1,
        vol_floor_quantile=0.4,
    )
    baseline = MeanReversion(timeframe="3m", rsi_n=5, vol_floor_window=0)

    warm_bar = {"window": high_df, "symbol": symbol, "timeframe": "3m"}
    strat.on_bar(warm_bar.copy())
    strat.on_bar(warm_bar.copy())
    baseline.on_bar(warm_bar.copy())

    baseline_bar = {"window": low_df, "symbol": symbol, "timeframe": "3m"}
    base_sig = baseline.on_bar(baseline_bar)
    assert base_sig is not None

    filtered_bar = {"window": low_df, "symbol": symbol, "timeframe": "3m"}
    filtered_sig = strat.on_bar(filtered_bar)
    assert filtered_sig is None
    assert filtered_bar["volatility_floor_bps"] is not None
    assert filtered_bar["volatility_bps"] < filtered_bar["volatility_floor_bps"]


def test_mean_reversion_dynamic_floor_reduces_backtest_fees(monkeypatch):
    high_prices = [100, 114, 92, 120, 88, 126, 84, 130, 82, 134]
    low_prices = [100.0, 100.2, 99.95, 100.15, 100.05, 100.12, 100.02, 100.08, 100.01, 100.06]
    closes = high_prices + low_prices
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.6 for c in closes],
            "low": [c - 0.6 for c in closes],
            "close": closes,
            "volume": [500.0] * len(closes),
        }
    )
    split_index = len(high_prices)

    def fake_rsi(df: pd.DataFrame, n: int) -> pd.Series:  # noqa: ANN001
        pattern = [70.0 if i % 2 == 0 else 30.0 for i in range(len(df))]
        return pd.Series(pattern, index=df.index)

    monkeypatch.setattr(mr, "rsi", fake_rsi)
    monkeypatch.setattr(MeanReversion, "auto_threshold", lambda self, series: (60, 40))

    fee_rate = 0.001
    symbol = "TEST"

    def run_strategy(vol_window: int) -> tuple[float, int, list[bool]]:
        risk = ClampRiskService()
        strat = MeanReversion(
            timeframe="3m",
            rsi_n=5,
            vol_floor_window=vol_window,
            vol_floor_min_periods=1,
            vol_floor_quantile=0.4,
            risk_service=risk,
        )
        total_fee = 0.0
        low_vol_signals = 0
        low_vol_flags: list[bool] = []
        for end in range(strat.rsi_n + 1, len(df) + 1):
            window = df.iloc[:end]
            bar = {"window": window, "symbol": symbol, "timeframe": "3m"}
            sig = strat.on_bar(bar)
            if end - 1 >= split_index and sig is not None:
                low_vol_signals += 1
                low_vol_flags.append(True)
            elif end - 1 >= split_index:
                low_vol_flags.append(False)
            if sig is None:
                continue
            price = float(window["close"].iloc[-1])
            qty = (
                strat.trade.get("qty")
                if getattr(strat, "trade", None)
                else float(sig.strength)
            )
            total_fee += abs(price * qty) * fee_rate
        return total_fee, low_vol_signals, low_vol_flags

    baseline_fee, baseline_low, baseline_flags = run_strategy(0)
    filtered_fee, filtered_low, filtered_flags = run_strategy(4)

    assert baseline_low > 0
    assert filtered_low < baseline_low
    assert filtered_fee < baseline_fee
    assert len(filtered_flags) == len(baseline_flags) == len(low_prices)
    assert sum(filtered_flags) == filtered_low
    assert sum(baseline_flags) == baseline_low
    assert all(flag is False for flag in filtered_flags[1:])
