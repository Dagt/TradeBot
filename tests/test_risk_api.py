import importlib

from fastapi.testclient import TestClient
from tradingbot.bus import EventBus
from tradingbot.core import Account
from tradingbot.risk.portfolio_guard import PortfolioGuard, GuardConfig
from tradingbot.risk.service import RiskService


def get_app():
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main.app


def test_risk_halt_publishes_and_auth(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    app = get_app()
    bus = EventBus()
    events = []
    bus.subscribe("control:halt", lambda msg: events.append(msg))
    app.state.bus = bus
    client = TestClient(app)

    resp = client.post("/risk/halt", json={"reason": "stop"})
    assert resp.status_code == 401

    resp = client.post("/risk/halt", json={"reason": "stop"}, auth=("u", "p"))
    assert resp.status_code == 200
    assert events == [{"reason": "stop"}]


def test_risk_reset_resets_and_auth(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    app = get_app()
    guard = PortfolioGuard(GuardConfig(total_cap_pct=1.0, per_symbol_cap_pct=1.0, venue="X"))
    rs = RiskService(guard, account=Account(float("inf")))
    rs.enabled = False
    rs.last_kill_reason = "dd"
    app.state.risk_service = rs
    client = TestClient(app)

    resp = client.post("/risk/reset")
    assert resp.status_code == 401

    resp = client.post("/risk/reset", auth=("u", "p"))
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk"]["enabled"] is True
    assert data["risk"]["last_kill_reason"] is None
    assert rs.enabled is True
