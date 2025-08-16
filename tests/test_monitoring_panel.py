from fastapi.testclient import TestClient

from monitoring.panel import app
from monitoring.metrics import (
    TRADING_PNL,
    SYSTEM_DISCONNECTS,
    MARKET_LATENCY,
    ORDER_LATENCY,
    MAKER_TAKER_RATIO,
    E2E_LATENCY,
    WS_FAILURES,
    STRATEGY_STATE,
    OPEN_POSITIONS,
)


def test_panel_endpoints_and_metrics():
    client = TestClient(app)

    TRADING_PNL.set(100)
    SYSTEM_DISCONNECTS.inc()
    MARKET_LATENCY.observe(0.5)
    OPEN_POSITIONS.clear()
    OPEN_POSITIONS.labels(symbol="BTCUSD").set(2)
    ORDER_LATENCY.clear()
    MAKER_TAKER_RATIO.clear()
    WS_FAILURES.clear()
    STRATEGY_STATE.clear()
    ORDER_LATENCY.labels(venue="test").observe(0.2)
    MAKER_TAKER_RATIO.labels(venue="test").set(1.5)
    E2E_LATENCY.observe(0.7)
    WS_FAILURES.labels(adapter="test").inc()
    STRATEGY_STATE.labels(strategy="alpha").set(1)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "TradeBot Monitoring" in resp.text

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "trading_pnl" in resp.text

    resp = client.get("/metrics/summary")
    data = resp.json()
    assert data["pnl"] == 100.0
    assert data["positions"] == {"BTCUSD": 2.0}
    assert data["disconnects"] == 1.0
    assert data["avg_market_latency_seconds"] == 0.5
    assert data["avg_order_latency_seconds"] == 0.2
    assert data["avg_maker_taker_ratio"] == 1.5
    assert data["avg_e2e_latency_seconds"] == 0.7
    assert data["ws_failures"] == 1.0
    assert data["strategy_states"] == {"alpha": 1.0}
