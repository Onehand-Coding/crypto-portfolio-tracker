import asyncio
import json
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
        return {"total_value_usd": 57.78, "holdings_df": pd.DataFrame()}

    mock_tracker.run_full_sync = fake_sync

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
async def test_runner_reports_core_errors_on_the_terminal_event(mock_tracker, tmp_path):
    """A sync that logs failures and then reports a bare "Sync complete" reads
    as total success. A failed fiat-rate lookup is exactly what once hid
    $50.41 of inflow, and it is logged at ERROR while the sync keeps going."""
    async def fake_sync():
        log = logging.getLogger("crypto_portfolio_tracker.price_enricher")
        log.error("HTTP error fetching exchange rate for PHPUSD=X")
        log.warning("Simple Earn rewards endpoint unavailable")
        return {"total_value_usd": 57.78, "holdings_df": pd.DataFrame()}

    mock_tracker.run_full_sync = fake_sync

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True

    messages = []
    async for event in runner.events():
        messages.append(event)
        if event["event"] == "complete":
            break

    complete = messages[-1]
    assert complete["error_count"] == 1
    assert complete["warning_count"] == 1
    # The ERROR must not have closed the stream: the sync was still running.
    assert complete["event"] == "complete"
    levels = [e.get("level") for e in messages if e["event"] == "progress"]
    assert "ERROR" in levels


@pytest.mark.asyncio
async def test_runner_refuses_concurrent_syncs(mock_tracker, tmp_path):
    async def slow_sync():
        await asyncio.sleep(0.2)
        return {"total_value_usd": 1.0, "holdings_df": pd.DataFrame()}

    mock_tracker.run_full_sync = slow_sync

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True
    assert runner.start() is False
    # Drain the slow run this test started: leaving it orphaned lets its
    # lowered INFO level outlive this test and pollute the core logger
    # for later tests.
    async for event in runner.events():
        if event["event"] in ("complete", "error"):
            break


@pytest.mark.asyncio
async def test_runner_writes_metrics_cache_on_success(mock_tracker, tmp_path):
    cache_file = tmp_path / "metrics.json"

    async def fake_sync():
        return {"total_value_usd": 57.78, "holdings_df": pd.DataFrame()}

    mock_tracker.run_full_sync = fake_sync

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


def test_post_sync_starts_a_real_sync_through_the_route(mock_tracker):
    """The route and the real SyncRunner together, which is what production
    runs. Every other test here either drives the runner directly from inside
    an async test -- where an event loop is already running -- or substitutes a
    fake runner. Neither exercises the seam, and the seam is where this broke:
    a plain `def` endpoint runs in a threadpool with no running event loop, so
    SyncRunner.start()'s get_running_loop() raised and every sync 500'd.
    """
    async def fake_sync():
        return {"total_value_usd": 57.78, "holdings_df": pd.DataFrame()}

    mock_tracker.run_full_sync = fake_sync

    response = TestClient(app).post("/api/sync")

    assert response.status_code == 200, response.text
    assert response.json() == {"status": "started"}


def test_post_sync_returns_409_when_already_running(mock_tracker, monkeypatch):
    from api.routes import sync as sync_route

    class AlwaysBusy:
        is_running = True

        def start(self):
            return False

    monkeypatch.setattr(sync_route, "get_sync_runner", lambda: AlwaysBusy())
    assert TestClient(app).post("/api/sync").status_code == 409


@pytest.mark.asyncio
async def test_runner_does_not_recompute_metrics_after_sync(mock_tracker, tmp_path):
    """run_full_sync already returns the metrics. Calling
    calculate_portfolio_metrics again repeats the full Binance and yfinance
    price enrichment for no gain."""
    from unittest.mock import AsyncMock

    mock_tracker.run_full_sync = AsyncMock(return_value={
        "total_value_usd": 57.78, "holdings_df": pd.DataFrame(),
    })
    mock_tracker.calculate_portfolio_metrics = AsyncMock()

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    runner.start()
    async for event in runner.events():
        if event["event"] in ("complete", "error"):
            break

    mock_tracker.calculate_portfolio_metrics.assert_not_called()


@pytest.mark.asyncio
async def test_runner_reports_error_when_tracker_construction_fails(tmp_path, monkeypatch):
    """CryptoPortfolioTracker.__init__ pings Binance and raises
    NetworkUnavailableError when offline. If that escapes _run, no error event
    is emitted and the SSE client waits forever on a sync that already died."""
    def boom():
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("api.sync_runner.get_tracker", boom)

    runner = SyncRunner(cache_path=tmp_path / "metrics.json")
    assert runner.start() is True

    events = []
    async for event in runner.events():
        events.append(event)
        if event["event"] in ("complete", "error"):
            break

    assert events[-1]["event"] == "error"
    assert "network unavailable" in events[-1]["message"]
    assert not (tmp_path / "metrics.json").exists()


def test_stream_route_emits_parseable_sse_frames(monkeypatch):
    """The browser's EventSource only recognises "data: {json}\\n\\n". The
    runner's dict events are covered above, but nothing asserted the wire
    format itself -- a change to the framing breaks the UI with no test
    failing anywhere.
    """
    class FakeRunner:
        def start(self):
            return True

        async def events(self):
            yield {"event": "progress", "message": "Fetching chunk 1 of 3", "level": "INFO"}
            yield {"event": "complete", "message": "Sync complete",
                   "error_count": 0, "warning_count": 0}

    monkeypatch.setattr("api.routes.sync.get_sync_runner", FakeRunner)

    with TestClient(app) as client:
        response = client.get("/api/sync/stream")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = [f for f in response.text.split("\n\n") if f.strip()]
    assert len(frames) == 2
    for frame in frames:
        assert frame.startswith("data: ")
        json.loads(frame[len("data: "):])

    assert json.loads(frames[-1][len("data: "):])["event"] == "complete"


@pytest.mark.asyncio
async def test_quiet_run_leaves_logger_level_untouched(mock_tracker, tmp_path):
    """Auto-syncs must not lower the core logger to INFO: with ~288 runs a
    day, chunk-progress INFO records would bury real warnings in the log."""
    core_logger = logging.getLogger("crypto_portfolio_tracker")
    entry_level = core_logger.level
    observed: dict = {}

    async def fake_sync():
        # Capture the level *during* the run: the pre-fix code lowers it
        # to INFO before this point, so this is what actually pins the
        # quiet behaviour (the post-run level is restored either way).
        observed["level"] = core_logger.level
        core_logger.info("chunk noise")
        return {}

    mock_tracker.run_full_sync = fake_sync

    runner = SyncRunner(cache_path=tmp_path / "m.json")
    try:
        assert runner.start(quiet=True) is True
        async for event in runner.events():
            if event["event"] in ("complete", "error"):
                break
        post_run_level = core_logger.level
    finally:
        core_logger.setLevel(entry_level)

    assert observed["level"] == logging.NOTSET
    assert post_run_level == logging.NOTSET
    assert (tmp_path / "m.json").exists()


@pytest.mark.asyncio
async def test_default_run_lowers_logger_level_during_sync(mock_tracker, tmp_path):
    """The default (manual) path lowers the core logger to INFO so
    chunk-progress records reach the run's handler. Paired with the quiet
    test above: together they pin that the quiet flag -- not an inversion --
    controls the level change."""
    core_logger = logging.getLogger("crypto_portfolio_tracker")
    entry_level = core_logger.level
    observed: dict = {}

    async def fake_sync():
        observed["level"] = core_logger.level
        return {}

    mock_tracker.run_full_sync = fake_sync

    runner = SyncRunner(cache_path=tmp_path / "m.json")
    try:
        assert runner.start() is True
        async for event in runner.events():
            if event["event"] in ("complete", "error"):
                break
        post_run_level = core_logger.level
    finally:
        core_logger.setLevel(entry_level)

    assert observed["level"] == logging.INFO
    assert post_run_level == entry_level


def _status_setup(mock_read_context, monkeypatch, age, config):
    """Point GET /api/sync/status at a stub cache with a fixed age."""
    import api.routes.sync as sync_route

    class StubCache:
        def __init__(self, path):
            self.path = path

        def age_seconds(self):
            return age

        def read(self):
            return {"_cached_at": 0} if age is not None else None

    monkeypatch.setattr(sync_route, "MetricsCache", StubCache)
    cm = mock_read_context.config_manager
    cm.is_testnet_mode = True
    cm.config = config
    return cm


def test_sync_status_reports_age_and_idle(mock_read_context, monkeypatch):
    _status_setup(mock_read_context, monkeypatch, age=100,
                  config={"automation": {"auto_sync": {"enabled": True,
                                                       "interval_minutes": 5}}})
    body = TestClient(app).get("/api/sync/status").json()
    assert body["is_running"] is False
    assert body["staleness"]["age_seconds"] == 100
    assert body["staleness"]["is_stale"] is False


def test_sync_status_stale_after_three_intervals(mock_read_context, monkeypatch):
    # Threshold is 3x the configured interval: 5 min -> stale past 900 s.
    _status_setup(mock_read_context, monkeypatch, age=901,
                  config={"automation": {"auto_sync": {"enabled": True,
                                                       "interval_minutes": 5}}})
    body = TestClient(app).get("/api/sync/status").json()
    assert body["staleness"]["is_stale"] is True


def test_sync_status_honours_custom_interval(mock_read_context, monkeypatch):
    # 2 min interval -> 360 s threshold, so 400 s is stale.
    _status_setup(mock_read_context, monkeypatch, age=400,
                  config={"automation": {"auto_sync": {"enabled": True,
                                                       "interval_minutes": 2}}})
    body = TestClient(app).get("/api/sync/status").json()
    assert body["staleness"]["is_stale"] is True


def test_sync_status_never_synced_is_stale(mock_read_context, monkeypatch):
    _status_setup(mock_read_context, monkeypatch, age=None, config={})
    body = TestClient(app).get("/api/sync/status").json()
    assert body["staleness"]["age_seconds"] is None
    assert body["staleness"]["is_stale"] is True


def test_sync_status_reports_running(mock_read_context, monkeypatch):
    import api.routes.sync as sync_route

    _status_setup(mock_read_context, monkeypatch, age=10, config={})

    class BusyRunner:
        is_running = True

    monkeypatch.setattr(sync_route, "get_sync_runner", lambda: BusyRunner())
    body = TestClient(app).get("/api/sync/status").json()
    assert body["is_running"] is True
