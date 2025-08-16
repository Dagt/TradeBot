import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from analysis.backtest_report import generate_report


def test_generate_report_basic():
    result = {
        "equity": 10.0,
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
    assert report["pnl"] == 10.0
    assert report["fill_rate"] == 1.5 / 2.0
    assert abs(report["slippage"] - 1.0) < 1e-9
