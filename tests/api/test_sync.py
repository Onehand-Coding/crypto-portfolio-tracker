import asyncio
import logging

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.sync_runner import SyncRunner


@pytest.mark.asyncio
async def test_runner_forwards_core_log_records_as_events(mock_tracker, tmp_path):
    async def fake_sync():
        logging.getLogger("crypto_portfolio_tracker.binance_fetcher").info(
            "Fetching chunk 1 of 3"
        )
        return True

    mock_tracker.run_full_sync = fake_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 57.78, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True

    messages = []
    async for event in runner.events():
        messages.append(event)
        if event["event"] == "complete":
            break

    assert any("chunk 1 of 3" in e.get("message", "") for e in messages)
    assert messages[-1]["event"] == "complete"


@pytest.mark.asyncio
async def test_runner_refuses_concurrent_syncs(mock_tracker, tmp_path):
    async def slow_sync():
        await asyncio.sleep(0.2)
        return True

    mock_tracker.run_full_sync = slow_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 1.0, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True
    assert runner.start() is False


@pytest.mark.asyncio
async def test_runner_writes_metrics_cache_on_success(mock_tracker, tmp_path):
    cache_file = tmp_path / "metrics.json"

    async def fake_sync():
        return True

    mock_tracker.run_full_sync = fake_sync
    mock_tracker.calculate_portfolio_metrics = _async_return({
        "total_value_usd": 57.78, "holdings_df": pd.DataFrame(),
    })

    runner = SyncRunner(cache_path=cache_file)
    runner.start()
    async for event in runner.events():
        if event["event"] == "complete":
            break

    from api.cache import MetricsCache
    assert MetricsCache(cache_file).read()["total_value_usd"] == 57.78


@pytest.mark.asyncio
async def test_runner_emits_error_event_and_leaves_cache_untouched(mock_tracker, tmp_path):
    cache_file = tmp_path / "metrics.json"

    async def failing_sync():
        raise RuntimeError("binance unreachable")

    mock_tracker.run_full_sync = failing_sync

    runner = SyncRunner(cache_path=cache_file)
    runner.start()

    events = []
    async for event in runner.events():
        events.append(event)
        if event["event"] in ("error", "complete"):
            break

    assert events[-1]["event"] == "error"
    assert "binance unreachable" in events[-1]["message"]
    assert not cache_file.exists()


def test_post_sync_returns_409_when_already_running(mock_tracker, monkeypatch):
    from api.routes import sync as sync_route

    class AlwaysBusy:
        is_running = True

        def start(self):
            return False

    monkeypatch.setattr(sync_route, "get_sync_runner", lambda: AlwaysBusy())
    assert TestClient(app).post("/api/sync").status_code == 409


def _async_return(value):
    async def _inner():
        return value
    return _inner
