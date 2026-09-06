"""The analysis runner must not stall the event loop while a run starts.

Regression: tracker construction pings Binance synchronously (~12s observed).
Built inline in _run, it wedged the loop, so no other request was answered
until it returned -- the UI's reload could not learn the run had started, and
every Run button sat dead for the whole stall, then flashed "Running…" briefly
at the end.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

import api.analysis_runner as runner_module
from api.analysis_runner import AnalysisRunner


class _StubCache:
    written = None

    def __init__(self, path):
        pass

    def write(self, payload):
        _StubCache.written = payload


@pytest.mark.asyncio
async def test_run_start_keeps_event_loop_responsive(monkeypatch):
    _StubCache.written = None

    def slow_get_tracker():
        time.sleep(3)  # synchronous network I/O, like the Binance ping
        return SimpleNamespace(config_manager=SimpleNamespace())

    async def fake_adapter(tracker, params=None):
        return {"ok": True}

    monkeypatch.setattr(runner_module, "get_tracker", slow_get_tracker)
    monkeypatch.setitem(runner_module.KINDS, "dca", fake_adapter)
    monkeypatch.setattr(runner_module, "MetricsCache", _StubCache)
    monkeypatch.setattr(
        runner_module, "analysis_cache_path", lambda cm, kind: "stub")

    runner = AnalysisRunner()
    assert runner.start("dca") is True
    await asyncio.sleep(0)  # let the task reach construction
    assert runner.is_running("dca") is True  # visible immediately

    # The loop must stay responsive while construction blocks in its thread:
    # half a second of loop time must cost well under the 3s stall.
    tick = time.monotonic()
    await asyncio.sleep(0.5)
    assert time.monotonic() - tick < 2.0

    await asyncio.wait_for(runner._tasks["dca"], timeout=15)
    assert _StubCache.written == {"ok": True}
