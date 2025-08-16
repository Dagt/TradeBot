import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tradingbot.analysis.backtest_report import generate_report


def test_generate_report_basic():
    result = {
        "equity": 10.5,
        "equity_curve": [
            10.0,
            9.8,
            10.2,
            10.1,
            10.4,
            10.3,
            10.7,
            10.5,
        ],
        "orders": [
            {
                "qty": 1.0,
                "filled": 1.0,
                "place_price": 100.0,
                "avg_price": 101.0,
                "side": "buy",
            },
            {
                "qty": 1.0,
                "filled": 0.5,
                "place_price": 100.0,
                "avg_price": 99.0,
                "side": "sell",
            },
        ],
    }
    report = generate_report(result)
    assert report["pnl"] == 10.5
    assert report["fill_rate"] == 1.5 / 2.0
    assert abs(report["slippage"] - 1.0) < 1e-9
    assert pytest.approx(report["sharpe"], rel=1e-9) == 4.523813861160344
    assert pytest.approx(report["sortino"], rel=1e-9) == 24.0067220169515
    assert pytest.approx(report["deflated_sharpe_ratio"], rel=1e-9) == 0.7784159008283134
