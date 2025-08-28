"""Integration test for backtest-db streaming API."""

import asyncio
import sys
from fastapi.testclient import TestClient

from tradingbot.apps.api.main import app


def _collect_events(chunks: list[str]):
    events = []
    current = None
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith('event: '):
                current = line.split('event: ', 1)[1]
            elif line.startswith('data: ') and current is not None:
                data = line.split('data: ', 1)[1]
                events.append((current, data))
    return events


def test_backtest_db_stream_status_and_end(monkeypatch):
    client = TestClient(app)
    orig_exec = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        assert 'backtest-db' in args
        cmd = [
            sys.executable,
            '-u',
            '-c',
            "import sys, time; print('running'); time.sleep(0.1)",
        ]
        return await orig_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_exec)

    payload = {
        'command': 'backtest-db --venue binance_spot --symbol BTC/USDT --strategy momentum --start 2021-01-01 --end 2021-01-02'
    }
    resp = client.post('/cli/start', json=payload, auth=('admin', 'admin'))
    assert resp.status_code == 200
    job_id = resp.json()['id']

    chunks = []
    with client.stream('GET', f'/cli/stream/{job_id}', auth=('admin', 'admin')) as stream:
        for raw in stream.iter_raw():
            text = raw.decode()
            chunks.append(text)
            if 'event: end' in text:
                break

    events = _collect_events(chunks)
    names = [ev for ev, _ in events]
    assert 'status' in names
    assert names[-1] == 'end'
