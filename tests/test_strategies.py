import math
import pandas as pd
import numpy as np
import yaml
from tradingbot.strategies.breakout_atr import BreakoutATR
from tradingbot.strategies.order_flow import OrderFlow
from tradingbot.strategies.mean_rev_ofi import MeanRevOFI
from tradingbot.strategies.breakout_vol import BreakoutVol
from tradingbot.core import Account, RiskManager as CoreRiskManager
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService
from tradingbot.execution.order_types import Order
import pytest
from hypothesis import given, strategies as st, settings
from tradingbot.data.features import keltner_channels, atr


@pytest.mark.parametrize("timeframe", ["1m"])
def test_breakout_atr_signals(breakout_df_buy, breakout_df_sell, timeframe):
    strat = BreakoutATR(ema_n=2, atr_n=2)

    bar_common = {
        "window": breakout_df_buy,
        "volatility": 0.0,
        "timeframe": timeframe,
        "symbol": "BTC/USDT",
    }
    sig_buy = strat.on_bar(bar_common)
    tf_mult = strat._tf_multiplier(timeframe)
    offset_frac = strat._offset_fraction(tf_mult)
    if tf_mult <= 3:
        ema_used = 15
        atr_used = 10
    elif tf_mult >= 30:
        ema_used = max(strat.ema_n, 30)
        atr_used = max(strat.atr_n, 20)
    else:
        ema_used = strat.ema_n
        atr_used = strat.atr_n
    atr_buy = atr(breakout_df_buy, atr_used).dropna().iloc[-1]
    upper, lower = keltner_channels(breakout_df_buy, ema_used, atr_used, strat.mult)
    last_close_buy = breakout_df_buy["close"].iloc[-1]
    assert sig_buy.side == "buy"
    level_buy = float(upper.iloc[-1])
    raw_offset_buy = abs(float(atr_buy) * offset_frac)
    abs_price_buy = abs(last_close_buy)
    min_step_buy = abs_price_buy * 0.0005
    book_offset_buy = abs_price_buy * 0.0003
    expected_buy = level_buy + min(
        max(last_close_buy - level_buy, 0.0),
        max(book_offset_buy, 0.0),
    )
    assert sig_buy.limit_price == pytest.approx(expected_buy)
    assert sig_buy.limit_price <= last_close_buy
    assert getattr(sig_buy, "post_only", False)
    assert sig_buy.metadata.get("post_only") is True
    assert sig_buy.metadata.get("initial_offset") == pytest.approx(
        abs(sig_buy.limit_price - level_buy)
    )
    expected_step_buy = max(
        min_step_buy,
        min(raw_offset_buy if raw_offset_buy > 0 else min_step_buy, abs_price_buy * 0.006),
    )
    assert sig_buy.metadata.get("step_offset") == pytest.approx(expected_step_buy)

    sig_sell = strat.on_bar(
        {
            "window": breakout_df_sell,
            "volatility": 0.0,
            "timeframe": timeframe,
            "symbol": "BTC/USDT",
        }
    )
    atr_sell = atr(breakout_df_sell, atr_used).dropna().iloc[-1]
    upper, lower = keltner_channels(breakout_df_sell, ema_used, atr_used, strat.mult)
    last_close_sell = breakout_df_sell["close"].iloc[-1]
    assert sig_sell.side == "sell"
    level_sell = float(lower.iloc[-1])
    abs_price_sell = abs(last_close_sell)
    min_step_sell = abs_price_sell * 0.0005
    book_offset_sell = abs_price_sell * 0.0003
    expected_sell = last_close_sell + max(min_step_sell, max(book_offset_sell, 0.0))
    assert sig_sell.limit_price == pytest.approx(expected_sell)
    assert sig_sell.limit_price >= last_close_sell
    assert getattr(sig_sell, "post_only", False)
    assert sig_sell.metadata.get("post_only") is True
    assert sig_sell.metadata.get("initial_offset") == pytest.approx(0.0)


@pytest.mark.parametrize("timeframe", ["1m"])
def test_breakout_atr_risk_service_handles_stop_and_size(breakout_df_buy, timeframe):
    account = Account(float("inf"))
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X")
    )
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutATR(ema_n=2, atr_n=2, **{"risk_service": svc})
    sig = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": timeframe, "symbol": "BTC/USDT"}
    )
    assert sig and sig.side == "buy"
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    tf_mult = strat._tf_multiplier(timeframe)
    if tf_mult <= 3:
        stop_mult = 1.5
    elif tf_mult >= 30:
        stop_mult = 2.0
    else:
        stop_mult = 1.5
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade["atr"], atr_mult=stop_mult)
    assert trade["stop"] == pytest.approx(expected_stop)


def test_breakout_atr_vector_signal(breakout_df_buy):
    data = breakout_df_buy.copy()
    data.loc[data.index[-1], "close"] += 10.0
    data["volume"] = 100
    entries, exits = BreakoutATR.signal(
        data, ema_n=2, atr_n=2, mult=1.0, volume_factor=0
    )
    assert entries.iloc[-1]
    assert not exits.iloc[-1]


@pytest.mark.asyncio
async def test_breakout_atr_limit_crosses_market_gets_fill(
    breakout_df_buy, paper_adapter
):
    strat = BreakoutATR(ema_n=2, atr_n=2)
    symbol = "BTC/USDT"
    sig = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": "1m", "symbol": symbol}
    )
    assert sig and sig.side == "buy"

    tf_mult = strat._tf_multiplier("1m")
    if tf_mult <= 3:
        ema_used = 15
        atr_used = 10
    elif tf_mult >= 30:
        ema_used = max(strat.ema_n, 30)
        atr_used = max(strat.atr_n, 20)
    else:
        ema_used = strat.ema_n
        atr_used = strat.atr_n
    atr_buy = atr(breakout_df_buy, atr_used).dropna().iloc[-1]
    upper, _ = keltner_channels(breakout_df_buy, ema_used, atr_used, strat.mult)
    naive_limit = float(upper.iloc[-1]) + float(atr_buy) * strat._offset_fraction(tf_mult)
    last_close = float(breakout_df_buy["close"].iloc[-1])
    assert naive_limit < last_close
    assert sig.limit_price < last_close
    assert getattr(sig, "post_only", False)

    paper_adapter.update_last_price(symbol, last_close)
    res = await paper_adapter.place_order(
        symbol,
        sig.side,
        "limit",
        qty=1.0,
        price=sig.limit_price,
        post_only=True,
    )
    assert res["status"] == "new"
    assert res["pending_qty"] == pytest.approx(1.0)


def test_breakout_atr_requote_progression(breakout_df_buy, breakout_df_sell):
    strat = BreakoutATR(ema_n=2, atr_n=2)
    symbol = "BTC/USDT"

    sig_buy = strat.on_bar(
        {"window": breakout_df_buy, "volatility": 0.0, "timeframe": "1m", "symbol": symbol}
    )
    assert sig_buy and sig_buy.side == "buy"
    order_buy = Order(symbol, sig_buy.side, "limit", 1.0, price=sig_buy.limit_price)
    res_buy = {"price": sig_buy.limit_price}
    strat._track_requote(order_buy)
    strat._reprice_order(order_buy, res_buy)
    first_buy = order_buy.price
    assert first_buy > sig_buy.limit_price
    strat._track_requote(order_buy)
    strat._reprice_order(order_buy, res_buy)
    second_buy = order_buy.price
    assert second_buy >= first_buy
    base_buy = sig_buy.metadata.get("base_price", 0.0)
    max_off_buy = sig_buy.metadata.get("max_offset", float("inf"))
    if math.isfinite(max_off_buy):
        assert second_buy - base_buy <= max_off_buy + 1e-9
        if math.isclose(second_buy, first_buy):
            assert math.isclose(second_buy - base_buy, max_off_buy, rel_tol=1e-6, abs_tol=1e-9)

    sig_sell = strat.on_bar(
        {"window": breakout_df_sell, "volatility": 0.0, "timeframe": "1m", "symbol": symbol}
    )
    assert sig_sell and sig_sell.side == "sell"
    order_sell = Order(symbol, sig_sell.side, "limit", 1.0, price=sig_sell.limit_price)
    res_sell = {"price": sig_sell.limit_price}
    strat._track_requote(order_sell)
    strat._reprice_order(order_sell, res_sell)
    first_sell = order_sell.price
    assert first_sell < sig_sell.limit_price
    strat._track_requote(order_sell)
    strat._reprice_order(order_sell, res_sell)
    second_sell = order_sell.price
    assert second_sell <= first_sell
    base_sell = sig_sell.metadata.get("base_price", 0.0)
    max_off_sell = sig_sell.metadata.get("max_offset", float("inf"))
    if math.isfinite(max_off_sell):
        assert base_sell - second_sell <= max_off_sell + 1e-9
        if math.isclose(second_sell, first_sell):
            assert math.isclose(base_sell - second_sell, max_off_sell, rel_tol=1e-6, abs_tol=1e-9)


def test_order_flow_signals():
    df_buy = pd.DataFrame(
        {
            "bid_qty": [1, 2, 3, 4, 10],
            "ask_qty": [5, 4, 3, 2, 1],
            "close": [100, 101, 100, 102, 101],
        }
    )
    df_sell = pd.DataFrame(
        {
            "bid_qty": [10, 4, 3, 2, 1],
            "ask_qty": [1, 2, 3, 4, 10],
            "close": [101, 100, 102, 99, 98],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=10.0, sell_threshold=10.0)
    sig_buy = strat.on_bar({"window": df_buy, "timeframe": "1m", "symbol": "X", "volatility": 0.0})
    assert sig_buy and sig_buy.side == "buy"
    assert sig_buy.limit_price == pytest.approx(df_buy["close"].iloc[-1])
    sig_sell = strat.on_bar({"window": df_sell, "timeframe": "1m", "symbol": "X", "volatility": 0.0})
    assert sig_sell and sig_sell.side == "sell"
    assert sig_sell.limit_price == pytest.approx(df_sell["close"].iloc[-1])


def test_mean_rev_ofi_signals():
    cfg = yaml.safe_load(
        """
ofi_window: 2
zscore_threshold: 0.5
vol_window: 2
vol_threshold: 0.9
"""
    )
    closes = [100, 110, 100, 105, 100, 101]
    df_sell = pd.DataFrame(
        {
            "bid_qty": [1, 2, 3, 4, 5, 10],
            "ask_qty": [5, 4, 3, 2, 1, 0],
            "close": closes,
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_sell = strat.on_bar({"window": df_sell, "volatility": 0.0, "timeframe": "3m"})
    assert sig_sell.side == "sell"

    df_buy = pd.DataFrame(
        {
            "bid_qty": [5, 4, 3, 2, 1, 0],
            "ask_qty": [1, 2, 3, 4, 5, 10],
            "close": closes,
        }
    )
    strat = MeanRevOFI(**cfg)
    sig_buy = strat.on_bar({"window": df_buy, "volatility": 0.0, "timeframe": "3m"})
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
    closes = list(range(1, 10)) + [50]
    df_buy = pd.DataFrame({"close": closes, "volume": [1.0] * len(closes)})
    account = Account(float("inf"))
    guard = PortfolioGuard(
        GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X")
    )
    svc = RiskService(
        guard,
        account=account,
        risk_per_trade=0.01,
        atr_mult=2.0,
        risk_pct=0.02,
    )
    svc.account.update_cash(1000.0)
    strat = BreakoutVol(lookback=2, **{"risk_service": svc})
    
    class DummyUpdater:
        def __init__(self, factor: float):
            self.factor = factor

        def update(self, value):
            return value * self.factor if value is not None else value

    class DummyRQ:
        def get(self, symbol, key, **kwargs):
            factor = 0.5 if key in {"std", "vol", "vol_bps"} else 1.0
            return DummyUpdater(factor)

    strat._rq = DummyRQ()
    sig = strat.on_bar({"window": df_buy, "volatility": 0.0, "symbol": "BTC/USDT"})
    assert sig and sig.side == "buy"
    last_price = df_buy["close"].iloc[-1]
    assert sig.limit_price > last_price
    trade = strat.trade
    assert trade is not None
    expected_qty = svc.calc_position_size(sig.strength, trade["entry_price"])
    assert trade["qty"] == pytest.approx(expected_qty)
    expected_stop = svc.initial_stop(trade["entry_price"], "buy", trade.get("atr"))
    assert trade["stop"] == pytest.approx(expected_stop)


def test_breakout_vol_more_signals_in_lower_timeframe():
    np.random.seed(0)
    prices = 100 + np.cumsum(np.random.normal(0, 0.03, 120))
    signals: dict[str, int] = {}
    for tf in ["1m", "5m"]:
        strat = BreakoutVol(lookback=10, timeframe=tf)
        sigs = 0
        for i in range(10, len(prices)):
            df = pd.DataFrame(
                {"close": prices[: i + 1], "volume": np.ones(i + 1, dtype=float)}
            )
            sig = strat.on_bar({"window": df})
            if sig and sig.side in {"buy", "sell"}:
                sigs += 1
        signals[tf] = sigs
    assert signals["1m"] > signals["5m"]


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_buy_property(start, inc):
    df = pd.DataFrame(
        {
            "bid_qty": [start + i * inc for i in range(3)],
            "ask_qty": [start - i * inc for i in range(3)],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None


@settings(deadline=None)
@given(start=st.floats(1, 10), inc=st.floats(0.1, 5))
def test_order_flow_sell_property(start, inc):
    df = pd.DataFrame(
        {
            "bid_qty": [start - i * inc for i in range(3)],
            "ask_qty": [start + i * inc for i in range(3)],
        }
    )
    strat = OrderFlow(window=3, buy_threshold=0.0, sell_threshold=0.0)
    sig = strat.on_bar({"window": df})
    assert sig is None
