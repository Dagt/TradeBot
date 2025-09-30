import math

import math

import pandas as pd

import tradingbot.strategies.momentum as momentum_module
from tradingbot.execution.order_types import Order
from tradingbot.strategies import STRATEGIES
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
    assert risk.calls and risk.calls[0].get("clamp") is False

    meta = sig.metadata
    anchor = meta["anchor_price"]
    assert math.isclose(anchor, best_bid, rel_tol=1e-9, abs_tol=1e-9)
    assert sig.limit_price <= anchor + 1e-9
    assert anchor - sig.limit_price <= bar["atr"] + 1e-9

    order = Order("X", sig.side, "limit", 1.0, price=sig.limit_price, post_only=sig.post_only)
    res = {"price": best_bid, "pending_qty": order.qty}
    for _ in range(3):
        action = strat.on_order_expiry(order, res)
        assert action == "re_quote"
        assert order.price <= anchor + 1e-9

    assert math.isclose(strat.trade["qty"], sig.strength, rel_tol=1e-9, abs_tol=1e-9)


def _breakout_window() -> pd.DataFrame:
    base = 200.0
    closes = [base - 0.2 * i for i in range(17)]
    closes.append(closes[-1] + 15.0)
    closes.append(closes[-1] + 3.0)
    highs = [c + 0.8 for c in closes]
    lows = [c - 0.9 for c in closes]
    lows[-1] = closes[-1] - 4.2
    highs[-1] = closes[-1] + 1.2
    bids = [c - 0.05 for c in closes]
    asks = [c + 0.05 for c in closes]
    volumes = [25.0] * 17 + [120.0, 80.0]
    return pd.DataFrame(
        {
            "timestamp": list(range(len(closes))),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "bid": bids,
            "ask": asks,
        }
    )


def _legacy_limit(side, price, anchor, atr_val, bar):  # noqa: ANN001
    limit_span = max(price * 0.001, atr_val * 0.5)
    limit_span = max(limit_span, abs(price - anchor))
    limit_span = max(limit_span, price * 0.0005)
    if not math.isfinite(limit_span) or limit_span <= 0:
        limit_span = max(abs(price) * 0.0005, 1e-6)

    if side == "buy":
        base_price = max(0.0, anchor - limit_span)
    else:
        base_price = anchor + limit_span

    initial_offset = max(limit_span * 0.4, price * 0.0003, atr_val * 0.25)
    initial_offset = min(initial_offset, limit_span)
    step_offset = max(limit_span * 0.25, price * 0.0002)
    step_offset = min(step_offset, limit_span)
    maker_initial = max(price * 0.0002, min(initial_offset * 0.5, limit_span))

    direction = 1.0 if side == "buy" else -1.0
    limit_price = base_price + direction * initial_offset
    if side == "buy":
        limit_price = min(limit_price, anchor)
    else:
        limit_price = max(limit_price, anchor)

    meta = {
        "base_price": base_price,
        "limit_offset": abs(limit_span),
        "initial_offset": abs(initial_offset),
        "offset_step": abs(step_offset),
        "max_offset": abs(limit_span),
        "maker_initial_offset": abs(maker_initial),
        "maker_patience": 1,
        "step_mult": 0.5,
        "chase": True,
        "post_only": True,
        "anchor_price": anchor,
    }
    return limit_price, meta


def test_momentum_breakout_backtest_improves_fills(monkeypatch):
    monkeypatch.setattr(momentum_module, "MIN_BARS", 3)

    df = _breakout_window()
    bar = {
        "window": df.iloc[:-1],
        "timeframe": "1m",
        "symbol": "X",
        "volatility": 0.0,
        "bid": float(df["bid"].iloc[-2]),
        "ask": float(df["ask"].iloc[-2]),
        "tick_size": 0.01,
    }

    strat = Momentum(
        fast_ema=2,
        slow_ema=4,
        rsi_n=3,
        atr_n=3,
        roc_n=1,
        vol_window=4,
        min_volume=0,
        min_volatility=0,
        use_rsi=False,
    )
    sig = strat.on_bar(bar)
    assert sig is not None and sig.side == "buy"
    anchor = sig.metadata["anchor_price"]
    assert anchor - sig.limit_price <= bar["atr"] + 1e-9

    orig_cls = momentum_module.Momentum

    class BacktestMomentum(orig_cls):
        name = "momentum"

        def __init__(self, **kwargs):
            kwargs.setdefault("fast_ema", 2)
            kwargs.setdefault("slow_ema", 4)
            kwargs.setdefault("rsi_n", 3)
            kwargs.setdefault("atr_n", 3)
            kwargs.setdefault("roc_n", 1)
            kwargs.setdefault("vol_window", 4)
            kwargs.setdefault("min_volume", 0)
            kwargs.setdefault("min_volatility", 0)
            kwargs.setdefault("use_rsi", False)
            super().__init__(**kwargs)

        def on_bar(self, bar):  # noqa: D401
            window = bar.get("window")
            if isinstance(window, pd.DataFrame) and not window.empty:
                if "bid" in window.columns:
                    bar.setdefault("bid", float(window["bid"].iloc[-1]))
                if "ask" in window.columns:
                    bar.setdefault("ask", float(window["ask"].iloc[-1]))
            bar.setdefault("tick_size", 0.01)
            return super().on_bar(bar)

    monkeypatch.setitem(STRATEGIES, "momentum", BacktestMomentum)

    from tradingbot.backtesting.engine import EventDrivenBacktestEngine

    new_engine = EventDrivenBacktestEngine(
        {"X": df},
        [("momentum", "X")],
        latency=0,
        window=17,
        verbose_fills=True,
        exchange_configs={"default": {"tick_size": 0.01}},
        risk_pct=0.01,
    )
    new_res = new_engine.run()
    new_orders = new_res["orders"]
    assert new_orders
    breakout_low = float(df["low"].iloc[-1])
    new_fills = sum(1 for order in new_orders if breakout_low <= order["place_price"] + 1e-9)

    legacy_helper = _legacy_limit
    monkeypatch.setattr(momentum_module, "_configure_limit", legacy_helper)

    legacy_engine = EventDrivenBacktestEngine(
        {"X": df},
        [("momentum", "X")],
        latency=0,
        window=17,
        verbose_fills=True,
        exchange_configs={"default": {"tick_size": 0.01}},
        risk_pct=0.01,
    )
    legacy_res = legacy_engine.run()
    legacy_orders = legacy_res["orders"]
    legacy_fills = sum(1 for order in legacy_orders if breakout_low <= order["place_price"] + 1e-9)

    assert new_fills > legacy_fills
