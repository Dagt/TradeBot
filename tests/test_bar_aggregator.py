from datetime import datetime, timezone

from tradingbot.live.runner import BarAggregator


def test_bar_aggregator_4h_alignment():
    agg = BarAggregator("4h")
    first_ts = datetime(2023, 1, 1, 5, 30, tzinfo=timezone.utc)
    agg.on_trade(first_ts, 100.0, 1.0)
    assert agg.current_period == datetime(2023, 1, 1, 4, 0, tzinfo=timezone.utc)

    second_ts = datetime(2023, 1, 1, 9, 0, tzinfo=timezone.utc)
    closed = agg.on_trade(second_ts, 110.0, 1.0)

    assert closed.ts_open == datetime(2023, 1, 1, 4, 0, tzinfo=timezone.utc)
    assert agg.current_period == datetime(2023, 1, 1, 8, 0, tzinfo=timezone.utc)
