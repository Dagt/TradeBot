import asyncio
import importlib
import os
import sys
import time


def get_module():
    sys.path.insert(0, 'src')
    import tradingbot.apps.api.main as main
    importlib.reload(main)
    return main


async def _run_backtest_process() -> asyncio.subprocess.Process:
    repo_root = os.getcwd()
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{repo_root}/src" + os.pathsep + env.get('PYTHONPATH', '')
    cmd = [
        sys.executable,
        '-u',
        '-m',
        'tradingbot.cli',
        'backtest',
        'data/examples/btcusdt_3m.csv',
        '--symbol',
        'BTC/USDT',
        '--strategy',
        'momentum',
    ]
    return await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


def _collect_events(chunks: list[str]) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    event = None
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith('event: '):
                event = line.split('event: ', 1)[1]
            elif line.startswith('data: ') and event is not None:
                data = line.split('data: ', 1)[1]
                events.append((event, data))
    return events


def test_stream_process_stops_after_end():
    main = get_module()

    async def run():
        proc = await _run_backtest_process()
        start = time.perf_counter()
        chunks: list[str] = []
        async for chunk in main._stream_process(proc, 'job', None, start):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())
    events = _collect_events(chunks)
    assert events, 'no events emitted'
    end_idx = next(i for i, (ev, _) in enumerate(events) if ev == 'end')
    assert all(ev != 'heartbeat' for ev, _ in events[end_idx + 1 :])

