"""Runs a full sync in the background and streams its progress.

The core already logs per-chunk progress while fetching 30-day windows.
Rather than adding callbacks to the core, this attaches a logging handler
for the duration of the run and forwards records to a queue. The core is
not modified.
"""

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Optional

from api.cache import MetricsCache
from api.deps import get_tracker

CORE_LOGGER = "crypto_portfolio_tracker"


class _QueueHandler(logging.Handler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.queue = queue
        self.loop = loop
        # Counted so the terminal event can say the sync finished with
        # failures in it. A sync that logs "HTTP error fetching exchange rate"
        # and then reports a bare "Sync complete" reads as total success, and
        # a failed fiat lookup is exactly what once hid $50.41 of inflow.
        self.error_count = 0
        self.warning_count = 0

    def emit(self, record: logging.LogRecord) -> None:
        # Serialised by Handler.handle's lock, so these increments are safe
        # despite arriving from the chunk-fetching worker threads.
        if record.levelno >= logging.ERROR:
            self.error_count += 1
        elif record.levelno >= logging.WARNING:
            self.warning_count += 1
        # Stays "progress" even for an ERROR record: "error" is a terminal
        # event that closes the stream, and a single failed price lookup must
        # not abort a sync that is still running. The level rides along so the
        # UI can render it as the failure it is.
        event = {"event": "progress", "message": record.getMessage(),
                 "level": record.levelname}
        self.loop.call_soon_threadsafe(self.queue.put_nowait, event)


class SyncRunner:
    def __init__(self, cache_path: Optional[Path] = None):
        self._cache_path = cache_path
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> bool:
        """Begin a sync. Returns False if one is already in flight."""
        if self.is_running:
            return False
        self._queue = asyncio.Queue()
        # get_running_loop, not get_event_loop: the latter is deprecated on
        # Python 3.10+ and this project runs 3.12.
        self._task = asyncio.get_running_loop().create_task(self._run())
        return True

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        logger = logging.getLogger(CORE_LOGGER)
        handler = _QueueHandler(self._queue, loop)
        previous_level = logger.level
        # The core logger has no explicit level, so it inherits root's
        # WARNING default and INFO-level chunk progress never reaches any
        # handler. Lower it for the run's duration only, then restore it.
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        def emit(event: dict) -> None:
            # Chunk fetching runs via asyncio.to_thread, so log records for
            # "progress" arrive through call_soon_threadsafe from a worker
            # thread. "complete"/"error" must queue behind any progress
            # records already scheduled that way, so they go through the
            # same call_soon_threadsafe path rather than a direct put --
            # otherwise a same-thread completion can overtake a still-
            # pending, cross-thread-scheduled progress callback.
            loop.call_soon_threadsafe(self._queue.put_nowait, event)

        try:
            emit({"event": "progress", "message": "Starting sync"})
            # Tracker construction (and cache-path resolution) happens inside
            # this try, not before it: CryptoPortfolioTracker.__init__ pings
            # Binance and can raise NetworkUnavailableError. If that happened
            # outside the try, the exception would escape _run entirely, no
            # error event would be enqueued, and an SSE client would block on
            # the queue forever even though the sync had already died.
            tracker = get_tracker()
            cache_path = self._cache_path
            if cache_path is None:
                from api.cache import cache_path_for
                cache_path = cache_path_for(tracker.config_manager)
            # run_full_sync already calls calculate_portfolio_metrics and returns
            # the result (portfolio_tracker.py:330-333). Calling it again would
            # repeat the full Binance + yfinance price enrichment -- the slowest
            # operation in the app -- for no gain.
            metrics = await tracker.run_full_sync()
            MetricsCache(cache_path).write(metrics)
            emit({
                "event": "complete",
                "message": "Sync complete",
                "error_count": handler.error_count,
                "warning_count": handler.warning_count,
            })
        except Exception as exc:  # surfaced to the UI, never swallowed
            emit({"event": "error", "message": str(exc)})
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous_level)

    async def events(self) -> AsyncIterator[dict]:
        # Single-consumer: this drains a shared queue, so two simultaneous
        # SSE clients split the events between them and each sees only a
        # partial stream. Acceptable for a single-user local tool.
        while True:
            event = await self._queue.get()
            yield event
            if event["event"] in ("complete", "error"):
                return


_runner: Optional[SyncRunner] = None


def get_sync_runner() -> SyncRunner:
    global _runner
    if _runner is None:
        _runner = SyncRunner()
    return _runner
