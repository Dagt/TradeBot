from tradingbot.utils.metrics import (
    FILL_COUNT,
    SLIPPAGE,
    RISK_EVENTS,
    ORDERS,
    SKIPS,
)
from tradingbot.live._metrics import infer_maker_flag


def test_fill_slippage_risk_metrics():
    # Ensure metrics start from a clean state
    FILL_COUNT.clear()
    SLIPPAGE.clear()
    RISK_EVENTS.clear()
    ORDERS._value.set(0)
    SKIPS._value.set(0)

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

    ORDERS.inc()
    sent_samples = list(ORDERS.collect())[0].samples
    sent_sample = [s for s in sent_samples if s.name == "orders_total"][0]
    assert sent_sample.value == 1.0

    SKIPS.inc()
    rej_samples = list(SKIPS.collect())[0].samples
    rej_sample = [s for s in rej_samples if s.name == "order_skips_total"][0]
    assert rej_sample.value == 1.0


def test_metrics_payload_maker_flag_inference():
    res = {"fee_type": "maker", "price": 100.0}
    maker_flag = infer_maker_flag(res, exec_price=100.0, base_price=100.0)
    payload = {
        "event": "fill",
        "side": "buy",
        "price": 100.0,
        "qty": 1.0,
        "fee": 0.0,
        "slippage_bps": 0.0,
        "maker": maker_flag,
    }
    assert payload["maker"] is True

    res["fee_type"] = "taker"
    payload["maker"] = infer_maker_flag(res, exec_price=100.0, base_price=100.0)
    assert payload["maker"] is False

    res.pop("fee_type")
    payload["maker"] = infer_maker_flag(res, exec_price=100.0, base_price=100.0)
    assert payload["maker"] is True

    payload["maker"] = infer_maker_flag(res, exec_price=99.5, base_price=100.0)
    assert payload["maker"] is False
