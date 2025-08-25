import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient

from tradingbot.exchanges import SUPPORTED_EXCHANGES


def test_ccxt_exchanges_includes_testnet(monkeypatch):
    monkeypatch.setenv("API_USER", "u")
    monkeypatch.setenv("API_PASS", "p")

    import tradingbot.apps.api.main as main
    importlib.reload(main)

    dummy_ccxt = SimpleNamespace(
        exchanges=[info["ccxt"] for info in SUPPORTED_EXCHANGES.values()]
    )
    monkeypatch.setattr(main, "ccxt", dummy_ccxt)

    client = TestClient(main.app)
    resp = client.get("/ccxt/exchanges", auth=("u", "p"))
    assert resp.status_code == 200
    expected = []
    for key in SUPPORTED_EXCHANGES:
        expected.append(key)
        expected.append(f"{key}_testnet")
    assert resp.json() == expected
