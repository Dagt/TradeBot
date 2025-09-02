import pandas as pd
import pytest

from tradingbot.backtesting.engine import EventDrivenBacktestEngine
from tradingbot.strategies import STRATEGIES


def test_stop_triggers_close(monkeypatch):
    class NoopStrategy:
        def on_bar(self, bar):
            return None

    monkeypatch.setitem(STRATEGIES, "noop", NoopStrategy)

    data = pd.DataFrame(
        {
            "timestamp": [0, 1],
            "open": [100.0, 90.0],
            "high": [100.0, 90.0],
            "low": [100.0, 90.0],
            "close": [100.0, 90.0],
            "volume": [1000, 1000],
        }
    )

    engine = EventDrivenBacktestEngine(
        {"SYM": data},
        [("noop", "SYM")],
        latency=0,
        window=1,
        verbose_fills=True,
    )

    svc = engine.risk[("noop", "SYM")]
    svc.update_position("default", "SYM", 1.0, entry_price=100.0)
    trade = svc.get_trade("SYM")
    trade["stop"] = 95.0
    monkeypatch.setattr(svc, "check_limits", lambda price: None)

    res = engine.run()
    assert res["orders"][0]["side"] == "sell"
    assert res["orders"][0]["qty"] == pytest.approx(1.0)
    assert len(res["fills"]) == 1
    assert res["fills"][0][0] == 1
