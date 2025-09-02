import sys
import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tradingbot.backtesting.engine import purged_kfold, walk_forward_optimize
from tradingbot.experiments.walk_forward import run_walk_forward_experiment
import mlflow


def test_purged_kfold_partitions():
    n = 20
    splits = purged_kfold(n, n_splits=4, embargo=0.1)
    assert len(splits) == 4
    embargo_size = int(n * 0.1)
    for train_idx, test_idx in splits:
        test_start, test_end = test_idx[0], test_idx[-1]
        embargo_range = set(
            range(max(0, test_start - embargo_size), min(n, test_end + embargo_size + 1))
        )
        assert embargo_range.isdisjoint(train_idx)


def test_walk_forward_optimize(tmp_path):
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

    params = [{"rsi_n": 2}, {"rsi_n": 3}]

    results = walk_forward_optimize(
        str(csv_path),
        "FOO/USDT",
        "momentum",
        params,
        n_splits=3,
        embargo=0.1,
    )

    assert len(results) == 3
    for i, rec in enumerate(results):
        assert rec["split"] == i
        assert "train_equity" in rec and "test_equity" in rec
        assert rec["params"] is not None


def test_walk_forward_mlflow(tmp_path):
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

    params = [{"rsi_n": 2}, {"rsi_n": 3}]

    mlflow.set_tracking_uri(str(tmp_path))
    exp_name = "wf_test"
    res = run_walk_forward_experiment(
        str(csv_path),
        "FOO/USDT",
        "momentum",
        params,
        n_splits=2,
        experiment=exp_name,
    )
    exp = mlflow.get_experiment_by_name(exp_name)
    assert exp is not None
    avg = sum(r["test_equity"] for r in res["results"]) / len(res["results"])
    assert res["avg_test_equity"] == avg
