from fastapi.testclient import TestClient
from monitoring.panel import app
import monitoring.panel as panel
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
    STRATEGY_ACTIONS,
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


def test_strategy_control_endpoints():
    client = TestClient(app)
    STRATEGY_STATE.clear()
    STRATEGY_ACTIONS.clear()

    resp = client.post("/strategies/foo/enable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    assert STRATEGY_STATE.labels(strategy="foo")._value.get() == 1.0
    assert (
        STRATEGY_ACTIONS.labels(strategy="foo", action="enable")._value.get() == 1.0
    )

    resp = client.post("/strategies/foo/params", json={"window": 10})
    assert resp.status_code == 200
    assert resp.json()["params"] == {"window": 10}
    assert (
        STRATEGY_ACTIONS.labels(strategy="foo", action="params")._value.get() == 1.0
    )

    resp = client.post("/strategies/foo/disable")
    assert resp.status_code == 200
    assert resp.json()["status"] == "stopped"
    assert STRATEGY_STATE.labels(strategy="foo")._value.get() == 0.0
    assert (
        STRATEGY_ACTIONS.labels(strategy="foo", action="disable")._value.get()
        == 1.0
    )


def test_alerts_endpoint(monkeypatch):
    client = TestClient(app)
    alerts_list = [
        {"labels": {"alertname": "RiskEvents"}},
        {"labels": {"alertname": "KillSwitchActive"}},
    ]
    monkeypatch.setattr(panel, "fetch_alerts", lambda: alerts_list)

    resp = client.get("/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_events"] is True
    assert data["kill_switch_active"] is True
    assert data["ws_disconnects"] is False
    assert data["alerts"] == alerts_list
