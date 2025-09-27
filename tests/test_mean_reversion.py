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
