import importlib

import os
from fastapi.testclient import TestClient


def get_app():
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main.app


def test_basic_auth(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    app = get_app()
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 401

    resp = client.get("/health", auth=("u", "p"))
    assert resp.status_code == 200
