import importlib
from pathlib import Path

import importlib
from fastapi.testclient import TestClient


def reload_app():
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main


def test_logs_endpoint(monkeypatch, tmp_path):
    log_file = tmp_path / "app.log"
    log_file.write_text("line1\nline2\n")
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    main = reload_app()
    main.settings.log_file = str(log_file)
    client = TestClient(main.app)
    resp = client.get("/logs", auth=("u", "p"))
    assert resp.status_code == 200
    assert "line1" in resp.json()["items"][0]


def test_orders_history(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")
    main = reload_app()
    client = TestClient(main.app)

    called = {}

    def fake_select(engine, limit=100, search=None, symbol=None, status=None):
        called.update(dict(limit=limit, search=search, symbol=symbol, status=status))
        return [
            {
                "ts": "2024-01-01T00:00:00Z",
                "strategy": "s",
                "symbol": "BTCUSDT",
                "side": "buy",
                "qty": 1,
                "px": 1,
                "status": "FILLED",
            }
        ]

    monkeypatch.setattr(main, "_CAN_PG", True)
    monkeypatch.setattr(main, "select_order_history", fake_select)

    resp = client.get("/orders/history?search=BTC", auth=("u", "p"))
    assert resp.status_code == 200
    assert called["search"] == "BTC"
    assert resp.json()["items"][0]["symbol"] == "BTCUSDT"
