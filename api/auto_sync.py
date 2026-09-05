"""Background scheduler for automatic portfolio syncs.

Safety property: this module only ever calls ``runner.start(quiet=True)``.
It cannot place orders by construction -- the runner only runs
``run_full_sync`` (history fetch + price enrichment + cache write), and
order execution lives in ``api/routes/execute.py``, which this module
never imports.
"""

import asyncio
import logging
from typing import Optional

from api.cache import MetricsCache, cache_path_for

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_MINUTES = 5
MIN_INTERVAL_MINUTES = 2
MAX_BACKOFF_SKIPS = 8


class AutoSyncScheduler:
    def __init__(self, config_manager, runner, tick_seconds: float = 30):
        self._cm = config_manager
        self._runner = runner
        self._tick_seconds = tick_seconds
        self._consecutive_failures = 0
        self._skips_owed = 0
        self._task: Optional[asyncio.Task] = None

    def _settings(self) -> tuple[bool, int]:
        """Read (enabled, minutes) fresh every call so interval changes
        apply without a restart."""
        try:
            auto = self._cm.config["automation"]["auto_sync"] or {}
        except (KeyError, TypeError, AttributeError):
            auto = {}
        if not isinstance(auto, dict):
            auto = {}
        enabled = bool(auto.get("enabled", False))
        try:
            minutes = int(auto.get("interval_minutes",
                                  DEFAULT_INTERVAL_MINUTES))
        except (TypeError, ValueError):
            minutes = DEFAULT_INTERVAL_MINUTES
        return enabled, max(minutes, MIN_INTERVAL_MINUTES)

    def _cache_age(self) -> Optional[float]:
        return MetricsCache(cache_path_for(self._cm)).age_seconds()

    async def _tick(self) -> None:
        enabled, minutes = self._settings()
        if not enabled:
            return
        if self._skips_owed > 0:
            self._skips_owed -= 1
            return
        age = self._cache_age()
        # A missing cache (fresh clone) counts as infinitely stale: only a
        # known-fresh cache suppresses the sync.
        if age is not None and age < minutes * 60:
            return
        if not self._runner.start(quiet=True):
            # A manual run or a previous auto run is already in flight.
            # That is a skip, never a failure.
            return
        # Drain the event queue to the terminal event. The drain exists to
        # observe the terminal outcome for backoff accounting: only the
        # terminal event tells us whether this run failed.
        # (SyncRunner.start() replaces the queue each run, so this is not
        # about preventing unbounded growth.)
        # No timeout on the drain: stop() cancels the run() task, which
        # aborts this await. Liveness therefore depends on stop() being
        # called on shutdown; without it a wedged runner would park the
        # scheduler here forever.
        async for event in self._runner.events():
            if event.get("event") == "error":
                self._consecutive_failures += 1
                self._skips_owed = min(self._consecutive_failures,
                                       MAX_BACKOFF_SKIPS)
                logger.warning("auto-sync run failed: %s",
                               event.get("message"))
            elif event.get("event") == "complete":
                # Partial-success policy: a complete terminal resets the
                # failure counter even if chunks logged errors. The run
                # refreshed prices and rewrote the cache; per-chunk errors
                # already surface via logs and the SSE error_count.
                self._consecutive_failures = 0

    async def run_once_for_testing(self) -> None:
        await self._tick()

    async def run(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception:
                # Deliberately Exception, not BaseException: CancelledError
                # is a BaseException, so it propagates through here and lets
                # stop() shut the loop down. Catching BaseException here
                # would swallow cancellation and break shutdown.
                logger.exception("auto-sync tick failed")
                self._consecutive_failures += 1
                self._skips_owed = min(self._consecutive_failures,
                                       MAX_BACKOFF_SKIPS)
            await asyncio.sleep(self._tick_seconds)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self.run())

    async def stop(self) -> None:
        # Read, don't pop: self._task stays set until the cancellation has
        # completed. A concurrent second stop() therefore observes the same
        # task and awaits the in-flight cancellation instead of returning
        # early, and an interleaved start() sees a non-done task and refuses
        # to spawn a second run() loop.
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # Only CancelledError is suppressed: it is the expected result
            # of the cancel() above. Any other exception is unexpected
            # (run() already logs per-tick Exceptions) and must propagate.
            pass
        finally:
            if self._task is task:
                self._task = None
