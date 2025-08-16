from fastapi.testclient import TestClient

from monitoring.panel import app
from monitoring.metrics import (
    TRADING_PNL,
    SYSTEM_DISCONNECTS,
    MARKET_LATENCY,
    ORDER_LATENCY,
    MAKER_TAKER_RATIO,
)


def test_panel_endpoints_and_metrics():
    client = TestClient(app)

    TRADING_PNL.set(100)
    SYSTEM_DISCONNECTS.inc()
    MARKET_LATENCY.observe(0.5)
    ORDER_LATENCY.clear()
    MAKER_TAKER_RATIO.clear()
    ORDER_LATENCY.labels(venue="test").observe(0.2)
    MAKER_TAKER_RATIO.labels(venue="test").set(1.5)

    resp = client.get("/")
    assert resp.status_code == 200
    assert "TradeBot Monitoring" in resp.text

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "trading_pnl" in resp.text

    resp = client.get("/metrics/summary")
    data = resp.json()
    assert data["pnl"] == 100.0
    assert data["disconnects"] == 1.0
    assert data["avg_order_latency_seconds"] == 0.2
    assert data["avg_maker_taker_ratio"] == 1.5
