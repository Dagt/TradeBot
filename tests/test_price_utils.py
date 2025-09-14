import pytest

from tradingbot.utils.price import limit_price_from_close


def test_limit_price_consistency_backtest_runner():
    close = 100.0
    tick = 0.01
    # Backtest: tick size 0 implies raw close
    assert limit_price_from_close("buy", close, 0.0) == pytest.approx(close)
    assert limit_price_from_close("sell", close, 0.0) == pytest.approx(close)
    # Runner: price already aligned to tick remains unchanged
    assert limit_price_from_close("buy", close, tick) == pytest.approx(close)
    assert limit_price_from_close("sell", close, tick) == pytest.approx(close)


def test_limit_price_tick_adjustment():
    close = 100.003
    tick = 0.01
    assert limit_price_from_close("buy", close, tick) == pytest.approx(100.0)
    assert limit_price_from_close("sell", close, tick) == pytest.approx(100.01)
