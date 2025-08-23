import importlib

from fastapi.testclient import TestClient

from tradingbot.cli.main import _AVAILABLE_VENUES


def get_app():
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main.app


def test_list_venues(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    app = get_app()
    client = TestClient(app)

    resp = client.get("/venues", auth=("u", "p"))
    assert resp.status_code == 200
    assert resp.json() == sorted(_AVAILABLE_VENUES)
