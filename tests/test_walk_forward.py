import sys
import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tradingbot.backtesting.engine import walk_forward_optimize


def test_walk_forward_optimize(tmp_path):
    # Create simple increasing price data
    n = 30
    df = pd.DataFrame(
        {
            "close": [float(i) for i in range(1, n + 1)],
            "high": [float(i) + 0.5 for i in range(1, n + 1)],
            "low": [float(i) - 0.5 for i in range(1, n + 1)],
            "volume": [100.0] * n,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    params = [{"rsi_n": 2, "rsi_threshold": 55}, {"rsi_n": 2, "rsi_threshold": 65}]

    results = walk_forward_optimize(
        str(csv_path),
        "FOO/USDT",
        "momentum",
        params,
        train_size=10,
        test_size=5,
    )

    assert len(results) == 4
    for i, rec in enumerate(results):
        assert rec["split"] == i
        assert "train_equity" in rec and "test_equity" in rec
        assert rec["params"] is not None
