import pytest
from tradingbot.strategies.arbitrage_triangular import TriangularArb


def test_triangular_arb_buy_signal():
    strat = TriangularArb()
    prices = {"bq": 100.0, "mq": 210.0, "mb": 2.0}
    sig = strat.on_bar({"prices": prices})
    assert sig.side == "buy"
    assert sig.strength == pytest.approx(0.05)
    assert sig.target_pct == pytest.approx(0.05)


def test_triangular_arb_sell_signal():
    strat = TriangularArb()
    prices = {"bq": 210.0, "mq": 100.0, "mb": 2.0}
    sig = strat.on_bar({"prices": prices})
    assert sig.side == "sell"
    assert sig.strength == pytest.approx(0.05)
    assert sig.target_pct == pytest.approx(0.05)


def test_triangular_arb_flat_when_incomplete_prices():
    strat = TriangularArb()
    sig = strat.on_bar({"prices": {"bq": 100.0, "mq": None, "mb": 2.0}})
    assert sig.side == "flat"
    assert sig.strength == pytest.approx(0.0)
    assert sig.target_pct == pytest.approx(0.0)


def test_triangular_arb_with_fees():
    strat = TriangularArb(taker_fee_bps=10.0)
    prices = {"bq": 100.0, "mq": 210.0, "mb": 2.0}
    sig = strat.on_bar({"prices": prices})
    assert sig.side == "buy"
    assert sig.strength == pytest.approx(0.046853, rel=1e-3)
    assert sig.target_pct == pytest.approx(0.046853, rel=1e-3)
