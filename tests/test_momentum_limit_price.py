import math

import pandas as pd

import tradingbot.strategies.momentum as momentum_module
from tradingbot.execution.order_types import Order
from tradingbot.strategies.momentum import Momentum


class DummyRiskService:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.min_order_qty = 0.0
        self.min_notional = 0.0
        self.pos = type("Pos", (), {"realized_pnl": 0.0, "qty": 0.0})()

    def calc_position_size(self, strength, price, **kwargs):  # noqa: ANN001
        self.calls.append(kwargs)
        return float(strength)

    @staticmethod
    def initial_stop(price, side, atr, atr_mult=None):  # noqa: ANN001
        atr_val = float(atr or 0.0)
        mult = 1.0
        if atr_mult is not None:
            try:
                mult = float(atr_mult)
            except (TypeError, ValueError):
                mult = 1.0
        offset = atr_val * mult
        if str(side).lower() == "buy":
            return float(price) - offset
        return float(price) + offset

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


def test_momentum_sets_maker_limit_and_clamps_size(monkeypatch):
    monkeypatch.setattr(momentum_module, "MIN_BARS", 2)
    df = pd.DataFrame(
        {
            "open": [1, 1, 1, 2, 1, 2],
            "high": [1.02, 1.02, 1.01, 2.02, 1.01, 2.02],
            "low": [0.98, 0.98, 0.99, 1.98, 0.99, 1.98],
            "close": [1, 1, 1, 2, 1, 2],
            "volume": [10.0] * 6,
        }
    )
    best_bid = 1.99
    best_ask = 2.01
    risk = DummyRiskService()
    strat = Momentum(
        fast_ema=2,
        slow_ema=4,
        rsi_n=3,
        atr_n=3,
        roc_n=1,
        vol_window=4,
        min_volume=0,
        min_volatility=0,
        risk_service=risk,
    )
    bar = {
        "window": df,
        "timeframe": "1m",
        "symbol": "X",
        "volatility": 0.0,
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
