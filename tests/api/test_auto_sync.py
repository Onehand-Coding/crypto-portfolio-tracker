"""Tests for the auto-sync scheduler (api/auto_sync.py)."""

import asyncio
from types import SimpleNamespace

import pytest

from api.auto_sync import MAX_BACKOFF_SKIPS, AutoSyncScheduler


class FakeRunner:
    """Minimal stand-in for SyncRunner: single-flight guard included."""

    def __init__(self, outcome="complete", running=False):
        self.starts = 0
        self.running = running
        self.outcome = outcome

    def start(self, quiet=False):
        assert quiet is True
        if self.running:
            return False
        self.starts += 1
        self.running = True
        return True

    async def events(self):
        # Mirror the real stream contract: progress events, then exactly one
        # terminal event, then return.
        yield {"event": "progress", "message": "Starting sync"}
        yield {"event": "progress", "message": "Fetched chunk",
               "level": "INFO"}
        self.running = False
        if self.outcome == "error":
            yield {"event": "error", "message": "boom"}
        else:
            yield {"event": "complete", "message": "Sync complete",
                   "error_count": 0, "warning_count": 0}


def make_scheduler(config, age, runner):
    cm = SimpleNamespace(config=config)
    sched = AutoSyncScheduler(cm, runner, tick_seconds=0)
    sched._cache_age = lambda: age
    return sched


def _enabled(interval_minutes=5):
    return {"automation": {"auto_sync": {"enabled": True,
                                        "interval_minutes": interval_minutes}}}


@pytest.mark.asyncio
async def test_disabled_by_default_does_nothing():
    runner = FakeRunner()
    sched = make_scheduler({"automation": {}}, None, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 0


@pytest.mark.asyncio
async def test_stale_cache_triggers_a_quiet_start():
    runner = FakeRunner()
    sched = make_scheduler(_enabled(5), 400, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 1


@pytest.mark.asyncio
async def test_fresh_cache_does_not_start():
    runner = FakeRunner()
    sched = make_scheduler(_enabled(5), 60, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 0


@pytest.mark.asyncio
async def test_missing_cache_starts_fresh_clone_path():
    runner = FakeRunner()
    sched = make_scheduler(_enabled(5), None, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 1


@pytest.mark.asyncio
async def test_in_flight_sync_is_a_skip_not_a_failure():
    runner = FakeRunner(running=True)
    sched = make_scheduler(_enabled(5), 9999, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 0
    assert sched._consecutive_failures == 0


@pytest.mark.asyncio
async def test_error_backs_off_and_complete_resets():
    runner = FakeRunner(outcome="error")
    sched = make_scheduler(_enabled(5), 9999, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 1
    assert sched._consecutive_failures == 1
    # Backoff: the next tick skips without starting (real tick expiry,
    # no hand-setting of _skips_owed).
    await sched._tick()
    assert runner.starts == 1
    # A successful run after expiry resets the counter.
    runner.outcome = "complete"
    await sched.run_once_for_testing()
    assert runner.starts == 2
    assert sched._consecutive_failures == 0


@pytest.mark.asyncio
async def test_backoff_expiry_uses_real_ticks_and_grows_linearly():
    runner = FakeRunner(outcome="error")
    sched = make_scheduler(_enabled(5), 9999, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 1
    assert sched._consecutive_failures == 1
    assert sched._skips_owed == 1
    # Owed skip consumes a real tick without starting.
    await sched._tick()
    assert runner.starts == 1
    assert sched._skips_owed == 0
    # Next real tick runs again and fails: linear growth to 2 owed.
    await sched.run_once_for_testing()
    assert runner.starts == 2
    assert sched._consecutive_failures == 2
    assert sched._skips_owed == 2
    # Both owed skips consume real ticks without starting.
    await sched._tick()
    assert runner.starts == 2
    assert sched._skips_owed == 1
    await sched._tick()
    assert runner.starts == 2
    assert sched._skips_owed == 0


@pytest.mark.asyncio
async def test_backoff_skips_cap_at_max():
    runner = FakeRunner(outcome="error")
    sched = make_scheduler(_enabled(5), 9999, runner)
    for _ in range(12):
        while sched._skips_owed > 0:
            before = runner.starts
            await sched._tick()
            assert runner.starts == before
        await sched.run_once_for_testing()
    assert sched._consecutive_failures == 12
    assert sched._skips_owed == MAX_BACKOFF_SKIPS == 8
    # Further failures never exceed the cap.
    while sched._skips_owed > 0:
        await sched._tick()
    await sched.run_once_for_testing()
    assert sched._skips_owed == 8


@pytest.mark.asyncio
async def test_interval_change_applies_without_restart():
    runner = FakeRunner()
    sched = make_scheduler(_enabled(5), 400, runner)
    await sched.run_once_for_testing()
    assert runner.starts == 1
    sched._cm.config["automation"]["auto_sync"]["interval_minutes"] = 1440
    sched._cache_age = lambda: 400
    await sched._tick()
    assert runner.starts == 1


@pytest.mark.asyncio
async def test_double_start_creates_one_task():
    runner = FakeRunner()
    cm = SimpleNamespace(config={"automation": {}})
    sched = AutoSyncScheduler(cm, runner, tick_seconds=0.01)
    try:
        sched.start()
        first = sched._task
        assert first is not None
        sched.start()
        assert sched._task is first
    finally:
        await sched.stop()
    assert sched._task is None
    assert first.done()


@pytest.mark.asyncio
async def test_stop_cancels_drain_and_returns():
    class BlockingRunner:
        def __init__(self):
            self.starts = 0

        def start(self, quiet=False):
            assert quiet is True
            self.starts += 1
            return True

        async def events(self):
            yield {"event": "progress", "message": "Starting sync"}
            await asyncio.sleep(3600)
            yield {"event": "complete", "message": "Sync complete",
                   "error_count": 0, "warning_count": 0}

    runner = BlockingRunner()
    sched = make_scheduler(_enabled(5), 9999, runner)
    sched._tick_seconds = 0.01
    sched.start()
    await asyncio.sleep(0.05)
    assert runner.starts == 1
    await asyncio.wait_for(sched.stop(), timeout=1)
    assert sched._task is None


@pytest.mark.asyncio
async def test_run_level_exception_increments_backoff():
    runner = FakeRunner()
    sched = make_scheduler({"automation": {}}, None, runner)
    sched._tick_seconds = 0.01
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("boom")

    sched._tick = flaky
    sched.start()
    try:
        await asyncio.sleep(0.05)
    finally:
        await sched.stop()
    assert calls >= 2
    assert sched._consecutive_failures == 1
    assert sched._skips_owed == 1
    assert sched._task is None
