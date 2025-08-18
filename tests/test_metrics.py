from tradingbot.utils.metrics import (
    FILL_COUNT,
    SLIPPAGE,
    RISK_EVENTS,
    ORDER_SENT,
    ORDER_REJECTS,
)


def test_fill_slippage_risk_metrics():
    # Ensure metrics start from a clean state
    FILL_COUNT.clear()
    SLIPPAGE.clear()
    RISK_EVENTS.clear()
    ORDER_SENT._value.set(0)
    ORDER_REJECTS._value.set(0)

    FILL_COUNT.labels(symbol="BTCUSDT", side="buy").inc()
    samples = list(FILL_COUNT.collect())[0].samples
    fill_sample = [s for s in samples if s.name == "order_fills_total" and s.labels["symbol"] == "BTCUSDT" and s.labels["side"] == "buy"][0]
    assert fill_sample.value == 1.0

    SLIPPAGE.labels(symbol="BTCUSDT", side="buy").observe(0.5)
    samples = list(SLIPPAGE.collect())[0].samples
    count_sample = [s for s in samples if s.name == "order_slippage_bps_count" and s.labels["symbol"] == "BTCUSDT" and s.labels["side"] == "buy"][0]
    sum_sample = [s for s in samples if s.name == "order_slippage_bps_sum" and s.labels["symbol"] == "BTCUSDT" and s.labels["side"] == "buy"][0]
    assert count_sample.value == 1.0
    assert sum_sample.value == 0.5

    RISK_EVENTS.labels(event_type="limit_breach").inc()
    samples = list(RISK_EVENTS.collect())[0].samples
    risk_sample = [s for s in samples if s.name == "risk_events_total" and s.labels["event_type"] == "limit_breach"][0]
    assert risk_sample.value == 1.0

    ORDER_SENT.inc()
    sent_samples = list(ORDER_SENT.collect())[0].samples
    sent_sample = [s for s in sent_samples if s.name == "order_sent_total"][0]
    assert sent_sample.value == 1.0

    ORDER_REJECTS.inc()
    rej_samples = list(ORDER_REJECTS.collect())[0].samples
    rej_sample = [s for s in rej_samples if s.name == "order_rejects_total"][0]
    assert rej_sample.value == 1.0
