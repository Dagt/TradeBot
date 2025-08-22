import importlib

from fastapi.testclient import TestClient


def get_app():
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main.app


def test_venue_kinds_endpoint(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    app = get_app()
    client = TestClient(app)

    resp = client.get("/venues/binance_spot/kinds", auth=("u", "p"))
    assert resp.status_code == 200
    data = resp.json()
    assert "trades" in data["kinds"]

    resp = client.get("/venues/unknown/kinds", auth=("u", "p"))
    assert resp.status_code == 404
