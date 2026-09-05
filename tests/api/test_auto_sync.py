"""Tests for the auto-sync scheduler (api/auto_sync.py)."""

from types import SimpleNamespace

import pytest

from api.auto_sync import AutoSyncScheduler


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
        self.running = False
        yield {"event": self.outcome}


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
    # Backoff: the next tick skips without starting.
    await sched._tick()
    assert runner.starts == 1
    # Simulate backoff expiry; a successful run resets the counter.
    runner.outcome = "complete"
    sched._skips_owed = 0
    await sched.run_once_for_testing()
    assert runner.starts == 2
    assert sched._consecutive_failures == 0


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
