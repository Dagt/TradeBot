import importlib

from fastapi.testclient import TestClient


def reload_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main


def test_api_secret_crud(tmp_path, monkeypatch):
    main = reload_app(tmp_path, monkeypatch)
    client = TestClient(main.app)

    resp = client.post("/secrets/FOO", json={"value": "bar"}, auth=("u", "p"))
    assert resp.status_code == 200

    resp = client.get("/secrets/FOO", auth=("u", "p"))
    assert resp.status_code == 200
    assert resp.json()["value"] == "bar"

    resp = client.delete("/secrets/FOO", auth=("u", "p"))
    assert resp.status_code == 200


def test_bots_html_has_no_secrets_section(tmp_path, monkeypatch):
    main = reload_app(tmp_path, monkeypatch)
    client = TestClient(main.app)

    resp = client.get("/static/bots.html", auth=("u", "p"))
    assert resp.status_code == 200
    assert "Secrets" not in resp.text
