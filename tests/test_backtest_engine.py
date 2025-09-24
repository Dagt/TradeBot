import math
import logging

import numpy as np
import pandas as pd
import pytest
try:  # pragma: no cover - optional dependency
    import vectorbt as vbt  # type: ignore
except Exception:  # pragma: no cover
    vbt = None
from types import SimpleNamespace

from tradingbot.backtest.event_engine import (
    EventDrivenBacktestEngine,
    SlippageModel,
    run_backtest_csv,
)
from tradingbot.strategies import STRATEGIES
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService
from tradingbot.core import Account



class DummyStrategy:
    name = "dummy"

    def __init__(self, **kwargs):
        self.i = 0

    def on_bar(self, bar):
        self.i += 1
        side = "buy" if self.i % 2 == 0 else "sell"
        return SimpleNamespace(side=side, strength=1.0)


def _make_csv(tmp_path):
    rng = pd.date_range("2021-01-01", periods=50, freq="T")
    price = np.linspace(100, 150, num=50)
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_pnl_with_and_without_slippage(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    strategies = [("dummy", "SYM")]
    data = {"SYM": str(csv_path)}

    no_slip = run_backtest_csv(data, strategies, latency=1, window=1, slippage=None)
    with_slip = run_backtest_csv(
        data, strategies, latency=1, window=1, slippage=SlippageModel(volume_impact=10.0)
    )

    assert no_slip["equity"] >= with_slip["equity"]


def test_fills_csv_export(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    strategies = [("dummy", "SYM")]
    data = {"SYM": str(csv_path)}
    out = tmp_path / "fills.csv"
    run_backtest_csv(data, strategies, latency=1, window=1, fills_csv=str(out))
    assert out.exists()
    df = pd.read_csv(out)
    assert not df.empty
    assert list(df.columns) == [
        "timestamp",
        "reason",
        "side",
        "price",
        "qty",
        "strategy",
        "symbol",
        "exchange",
        "fee_cost",
        "slippage_pnl",
        "realized_pnl",
        "realized_pnl_total",
        "equity_after",
    ]
    # Comprobar cálculo de fee
    expected_fee = df["price"] * df["qty"] * 0.001
    assert np.allclose(df["fee_cost"], expected_fee)
    # Comprobar que realized_pnl coincide con RiskService
    svc = RiskService(PortfolioGuard(GuardConfig(venue="test")), account=Account(float("inf")))
    expected = []
    for row in df.itertuples():
        prev = svc.pos.realized_pnl
        svc.add_fill(row.side, row.qty, row.price)
        delta = svc.pos.realized_pnl - prev
        expected.append(delta - row.fee_cost - row.slippage_pnl)
    assert np.allclose(df["realized_pnl"], expected)
    assert np.allclose(df["realized_pnl"].cumsum(), df["realized_pnl_total"])


def test_realized_pnl_includes_slippage(monkeypatch):
    class LimitStrategy:
        def __init__(self, risk_service=None):
            self.sent = False

        def on_bar(self, _):
            if self.sent:
                return None
            self.sent = True
            return SimpleNamespace(side="buy", strength=1.0, limit_price=101.0)

    monkeypatch.setitem(STRATEGIES, "limit_slippage", LimitStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("limit_slippage", "SYM")],
        latency=1,
        window=1,
        verbose_fills=True,
    )
    res = engine.run()

    order_summary = res["orders"][0]

    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )

    assert len(fills) == 2

    first_fill = fills.iloc[0]
    assert first_fill.reason == "order"
    assert first_fill.side == "buy"
    assert first_fill.slippage_pnl < 0
    assert order_summary["place_price"] > first_fill.price

    first_fee = first_fill.fee_cost
    assert first_fill.realized_pnl == pytest.approx(
        -(first_fee + first_fill.slippage_pnl)
    )
    assert first_fill.realized_pnl_total == pytest.approx(
        -(first_fee + first_fill.slippage_pnl)
    )

    final_fill = fills.iloc[-1]
    assert final_fill.reason == "liquidation"
    assert final_fill.side == "sell"

    final_fee = final_fill.fee_cost
    assert final_fill.realized_pnl == pytest.approx(-final_fee)
    assert final_fill.realized_pnl_total == pytest.approx(
        -(first_fee + final_fee + first_fill.slippage_pnl)
    )

    assert res["slippage"] == pytest.approx(first_fill.slippage_pnl)


def test_post_only_maker_has_zero_slippage(monkeypatch):
    class PostOnlyStrategy:
        def __init__(self, risk_service=None):
            self.sent = False

        def on_bar(self, _):
            if self.sent:
                return None
            self.sent = True
            return SimpleNamespace(
                side="buy",
                strength=1.0,
                post_only=True,
                limit_price=100.0,
            )

    monkeypatch.setitem(STRATEGIES, "post_only_maker", PostOnlyStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("post_only_maker", "SYM")],
        latency=1,
        window=1,
        verbose_fills=True,
    )
    res = engine.run()

    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )

    order_fills = fills[fills["reason"] == "order"]
    assert not order_fills.empty
    assert np.allclose(order_fills["slippage_pnl"], 0.0)

    liq_fills = fills[fills["reason"] == "liquidation"]
    assert not liq_fills.empty
    assert np.allclose(liq_fills["slippage_pnl"], 0.0)

    assert res["slippage"] == pytest.approx(0.0)

def test_spot_long_only_enforced(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=4, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    class FlipStrategy:
        def __init__(self):
            self.i = 0

        def on_bar(self, bar):
            self.i += 1
            side = "buy" if self.i % 2 == 1 else "sell"
            return SimpleNamespace(side=side, strength=1.0)

    monkeypatch.setitem(STRATEGIES, "flip", FlipStrategy)
    strategies = [("flip", "SYM")]
    data = {"SYM": str(path)}
    cfg = {"default": {"market_type": "spot"}}
    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs=cfg,
        initial_equity=1000.0,
        verbose_fills=True,
    )
    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )
    assert fills.loc[fills.side == "sell", "qty"].max() <= fills.loc[fills.side == "buy", "qty"].max() + 1e-9
    # Compute running position to ensure it never goes negative
    pos = (
        fills.apply(lambda r: r.qty if r.side == "buy" else -r.qty, axis=1).cumsum()
    )
    assert (pos >= -1e-9).all()


def test_slippage_bps_reduces_realized_and_equity(monkeypatch):
    class FixedSlippage:
        def __init__(self, pct: float = 0.0) -> None:
            self.pct = pct

        def fill(
            self,
            side: str,
            qty: float,
            price: float,
            bar,
            queue_pos: float = 0.0,
            partial: bool = True,
        ) -> tuple[float, float, float]:
            adj = price * (1 + self.pct) if side == "buy" else price * (1 - self.pct)
            return float(adj), float(qty), 0.0

    class WideLimitStrategy:
        def __init__(self, risk_service=None):
            self.step = 0

        def on_bar(self, _):
            self.step += 1
            if self.step == 1:
                return SimpleNamespace(side="buy", strength=1.0, limit_price=1e9)
            if self.step == 5:
                return SimpleNamespace(side="sell", strength=1.0, limit_price=-1e9)
            return None

    monkeypatch.setitem(STRATEGIES, "wide_limit", WideLimitStrategy)

    data = pd.DataFrame(
        {
            "timestamp": range(10),
            "open": [100.0] * 10,
            "high": [100.0] * 10,
            "low": [100.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000] * 10,
        }
    )

    def _run(slippage_bps: float) -> tuple[pd.DataFrame, dict]:
        slip = FixedSlippage()
        engine = EventDrivenBacktestEngine(
            {"SYM": data},
            [("wide_limit", "SYM")],
            latency=1,
            window=1,
            slippage=slip,
            slippage_bps=slippage_bps,
            initial_equity=1_000_000.0,
            verbose_fills=True,
        )
        result = engine.run()
        fills_df = pd.DataFrame(
            result["fills"],
            columns=[
                "timestamp",
                "reason",
                "side",
                "price",
                "qty",
                "strategy",
                "symbol",
                "exchange",
                "fee_cost",
                "slippage_pnl",
                "realized_pnl",
                "realized_pnl_total",
                "equity_after",
            ],
        )
        return fills_df, result

    fills_no_slip, res_no_slip = _run(0.0)
    fills_with_slip, res_with_slip = _run(100.0)

    final_realized_no_slip = fills_no_slip["realized_pnl_total"].iloc[-1]
    final_realized_with_slip = fills_with_slip["realized_pnl_total"].iloc[-1]

    assert final_realized_with_slip < final_realized_no_slip
    assert res_with_slip["equity"] < res_no_slip["equity"]


def test_spot_short_signal_rejected(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=2, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    class AlwaysShort:
        def on_bar(self, bar):
            return SimpleNamespace(side="sell", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "shorty", AlwaysShort)
    strategies = [("shorty", "SYM")]
    data = {"SYM": str(path)}
    cfg = {"default": {"market_type": "spot"}}
    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs=cfg,
        initial_equity=1000.0,
        verbose_fills=True,
    )
    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )
    assert fills.empty


def test_exchange_configs_require_market_type(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=2, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    class BuyOnce:
        def on_bar(self, bar):
            return SimpleNamespace(side="buy", strength=1.0)

    monkeypatch.setitem(STRATEGIES, "buyonce", BuyOnce)
    strategies = [("buyonce", "SYM", "badex")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(
        data, strategies, latency=1, window=1, exchange_configs={"badex": {}}
    )
    assert "equity" in res

    strategies = [("buyonce", "SYM", "good_spot")]
    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs={"good_spot": {}},
    )
    assert "equity" in res


def test_spot_balances_never_negative(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    class Greedy:
        def __init__(self):
            self.i = 0

        def on_bar(self, bar):
            self.i += 1
            if self.i == 1:
                return SimpleNamespace(side="buy", strength=5.0)
            if self.i == 2:
                return SimpleNamespace(side="sell", strength=5.0)
            if self.i == 3:
                return SimpleNamespace(side="buy", strength=5.0)
            return None

    monkeypatch.setitem(STRATEGIES, "greedy", Greedy)
    strategies = [("greedy", "SYM", "venue_spot")]
    data = {"SYM": str(path)}
    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        exchange_configs={"venue_spot": {}},
        initial_equity=1000.0,
        verbose_fills=True,
    )
    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )
    cash = 1000.0
    base = 0.0
    for row in fills.itertuples():
        if row.side == "buy":
            cash -= row.price * row.qty + row.fee_cost
            base += row.qty
        else:
            cash += row.price * row.qty - row.fee_cost
            base -= row.qty
        assert cash >= -1e-9
        assert base >= -1e-9


def test_spot_venue_config_applied(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )

    class BuyOnce:
        sent = False

        def on_bar(self, bar):
            if not self.sent:
                self.sent = True
                return SimpleNamespace(side="buy", strength=1.0)
            return None

    monkeypatch.setitem(STRATEGIES, "buyonce", BuyOnce)
    eng = EventDrivenBacktestEngine(
        {"SYM": df},
        [("buyonce", "SYM", "spot_venue")],
        initial_equity=1000.0,
        exchange_configs={
            "spot_venue": {
                "market_type": "spot",
                "maker_fee_bps": 10,
                "taker_fee_bps": 10,
            }
        },
        latency=1,
        window=1,
        verbose_fills=True,
    )

    risk_service = eng.risk[("buyonce", "SYM")]
    assert risk_service.allow_short is False

    res = eng.run()
    fills = pd.DataFrame(
        res["fills"],
        columns=[
            "timestamp",
            "reason",
            "side",
            "price",
            "qty",
            "strategy",
            "symbol",
            "exchange",
            "fee_cost",
            "slippage_pnl",
            "realized_pnl",
            "realized_pnl_total",
            "equity_after",
        ],
    )
    assert math.isclose(
        fills.loc[0, "fee_cost"],
        fills.loc[0, "price"] * fills.loc[0, "qty"] * 0.001,
        rel_tol=1e-9,
    )


def test_run_vectorbt_basic():
    vbt_local = pytest.importorskip("vectorbt")
    from tradingbot.backtest.vectorbt_engine import run_vectorbt

    class MAStrategy:
        @staticmethod
        def signal(df, fast, slow):
            close = df["close"]
            fast_ma = vbt_local.MA.run(close, fast)
            slow_ma = vbt_local.MA.run(close, slow)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            return entries, exits

    price = pd.Series(np.sin(np.linspace(0, 10, 100)) + 10)
    data = pd.DataFrame({"close": price})
    params = {"fast": [2, 4], "slow": [8]}

    stats = run_vectorbt(data, MAStrategy, params)
    assert not stats.empty
    assert {"sharpe_ratio", "max_drawdown", "total_return"} <= set(stats.columns)
    assert list(stats.index.names) == ["fast", "slow"]


@pytest.mark.optional
def test_vectorbt_wrapper_param_sweep():
    vbt_local = pytest.importorskip("vectorbt")
    from tradingbot.backtesting.vectorbt_wrapper import run_parameter_sweep

    def ma_signal(close, window):
        ma = vbt_local.MA.run(close, window).ma
        entries = close > ma
        exits = close < ma
        return entries, exits

    price = pd.Series(np.linspace(1, 2, 50))
    data = pd.DataFrame({"close": price})
    params = {"window": [2, 4]}
    stats = run_parameter_sweep(data, ma_signal, params)
    assert not stats.empty
    assert list(stats.index.names) == ["window"]


class OneShotStrategy:
    name = "oneshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=1.0)


def test_l2_queue_partial_and_cancel(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=3, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "bid_size": [1.0, 1.0, 1.0],
            "ask_size": [0.2, 0.2, 0.4],
            "volume": 1000,
        }
    )
    path = tmp_path / "l2.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "oneshot", OneShotStrategy)
    strategies = [("oneshot", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(
        data,
        strategies,
        latency=1,
        window=1,
        slippage=SlippageModel(),
        use_l2=True,
        cancel_unfilled=True,
    )

    assert pytest.approx(res["orders"][0]["filled"], rel=1e-9) == 0.0
    assert len(res["fills"]) == 0


class HalfShotStrategy:
    name = "halfshot"

    def __init__(self) -> None:
        self.sent = False

    def on_bar(self, bar):
        if self.sent:
            return None
        self.sent = True
        return SimpleNamespace(side="buy", strength=0.5)


def test_funding_payment(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=3, freq="H")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "volume": 1000,
            "funding_rate": [0.0, 0.0, 0.01],
        }
    )
    path = tmp_path / "fund.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "halfshot", HalfShotStrategy)
    strategies = [("halfshot", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(
        data, strategies, latency=1, window=1, initial_equity=200.0
    )
    assert pytest.approx(res["funding"], rel=1e-9) == 1.0
    # El costo de la operación incluye la comisión por defecto (0.001)
    assert pytest.approx(res["equity"], rel=1e-9) == 198.8


class AlwaysBuyStrategy:
    name = "always"

    def on_bar(self, bar):
        return SimpleNamespace(side="buy", strength=1.0)


def test_stop_on_equity_depletion(tmp_path, monkeypatch, caplog):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "deplete.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "always", AlwaysBuyStrategy)
    strategies = [("always", "SYM")]
    data = {"SYM": str(path)}

    caplog.set_level(logging.WARNING)
    out = tmp_path / "fills.csv"
    res = run_backtest_csv(
        data, strategies, latency=1, window=1, initial_equity=0.0, fills_csv=str(out)
    )

    df = pd.read_csv(out)
    assert len(df) == 0
    assert any("Equity depleted" in m for m in caplog.messages)


def test_continue_with_positive_mtm(tmp_path, monkeypatch, caplog):
    rng = pd.date_range("2021-01-01", periods=5, freq="T")
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1000,
        }
    )
    path = tmp_path / "deplete.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "always", AlwaysBuyStrategy)
    strategies = [("always", "SYM")]
    data = {"SYM": str(path)}

    caplog.set_level(logging.WARNING)
    res = run_backtest_csv(
        data, strategies, latency=1, window=1, initial_equity=1.0
    )

    assert len(res["equity_curve"]) == len(df) + 1
    assert not any("Equity depleted" in m for m in caplog.messages)


def test_seed_reproducibility(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)
    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    strategies = [("dummy", "SYM")]
    data = {"SYM": str(csv_path)}

    seeds = [1, 7, 42]
    equities = []
    for s in seeds:
        res = run_backtest_csv(data, strategies, latency=1, window=1, seed=s)
        equities.append(res["equity"])

    base = equities[0]
    for eq in equities[1:]:
        assert abs(eq - base) <= abs(base) * 0.005


def test_sharpe_is_annualised(tmp_path, monkeypatch):
    rng = pd.date_range("2021-01-01", periods=10, freq="D")
    price = np.linspace(100, 110, num=10)
    df = pd.DataFrame(
        {
            "timestamp": rng.view("int64") // 10**9,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        }
    )
    path = tmp_path / "daily.csv"
    df.to_csv(path, index=False)

    monkeypatch.setitem(STRATEGIES, "dummy", DummyStrategy)
    strategies = [("dummy", "SYM")]
    data = {"SYM": str(path)}

    res = run_backtest_csv(data, strategies, latency=1, window=1)
    rets = pd.Series(res["equity_curve"]).diff().dropna()
    expected = (rets.mean() / rets.std()) * np.sqrt(252)
    assert res["sharpe"] == pytest.approx(expected)

def test_max_drawdown_zero_initial_equity(tmp_path, monkeypatch):
    csv_path = _make_csv(tmp_path)

    class NoTradeStrategy:
        name = "no_trade"

        def on_bar(self, bar):
            return None

    monkeypatch.setitem(STRATEGIES, "no_trade", NoTradeStrategy)
    strategies = [("no_trade", "SYM")]
    data = {"SYM": str(csv_path)}

    res = run_backtest_csv(data, strategies, latency=1, window=1, initial_equity=0.0)

    assert res["max_drawdown"] == 0.0
    assert not math.isnan(res["max_drawdown"])


