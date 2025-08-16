import numpy as np
from hypothesis import given, strategies as st, assume

from tradingbot.utils.performance import dsr


@given(st.lists(st.floats(-0.1, 0.1, allow_nan=False, allow_infinity=False), min_size=2))
def test_dsr_bounds(data):
    assume(np.std(data, ddof=1) > 0)
    try:
        value = dsr(data, num_trials=5)
    except ValueError:
        assume(False)
    assert 0.0 <= value <= 1.0


def test_dsr_ordering():
    rng = np.random.default_rng(0)
    r_pos = rng.normal(0.001, 0.01, size=500)
    r_neg = rng.normal(-0.001, 0.01, size=500)
    assert dsr(r_pos, num_trials=10) > dsr(r_neg, num_trials=10)
