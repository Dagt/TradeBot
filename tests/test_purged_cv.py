import sys
import pathlib
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tradingbot.backtesting.purged_cv import purged_splits, run_purged_cv
from tradingbot.analysis.backtest_report import generate_report


class DummyStrategy:
    def fit(self, data):
        # no training required for the dummy strategy
        pass

    def predict(self, data):
        # always hold a long position
        return pd.Series(1, index=data.index)


def test_purged_splits_basic():
    n = 20
    splits = list(purged_splits(n, n_splits=4, embargo=0.1))
    assert len(splits) == 4
    embargo_size = int(n * 0.1)
    for train_idx, test_idx in splits:
        test_start, test_end = test_idx[0], test_idx[-1]
        embargo_range = set(
            range(max(0, test_start - embargo_size), min(n, test_end + embargo_size + 1))
        )
        assert embargo_range.isdisjoint(train_idx)


def test_run_purged_cv_and_report():
    n = 30
    df = pd.DataFrame({"close": [1.1 ** i for i in range(n)]})
    result = run_purged_cv(df, DummyStrategy, {}, n_splits=3, embargo=0.1)
    expected = 1.1 ** 9  # each test window yields 9 returns of 10%
    assert pytest.approx(result["purged_cv"], rel=1e-12) == expected

    report = generate_report({"equity": 0.0, "orders": [], "purged_cv": result["purged_cv"]})
    assert pytest.approx(report["purged_cv"], rel=1e-12) == expected
