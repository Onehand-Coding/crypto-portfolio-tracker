"""Explicit, user-initiated sync. The only path that contacts Binance."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.cache import MetricsCache, cache_path_for
from api.deps import get_read_context
from api.routes.common import staleness_for
from api.schemas.system import SyncStatusResponse
from api.sync_runner import get_sync_runner

router = APIRouter(prefix="/api/sync", tags=["sync"])

# Stale means the sync is not keeping up with the cadence configured in
# Settings (default 5 min), not a fixed clock: with auto-sync running, an
# hour-old cache is a failure, not a number.
STALE_AFTER_INTERVALS = 3


@router.post("")
async def start_sync() -> dict:
    # Must be async. FastAPI runs a plain `def` endpoint in a threadpool, and
    # SyncRunner.start() calls asyncio.get_running_loop() to schedule the sync
    # task -- there is no running loop in a worker thread, so every sync 500'd
    # with "no running event loop". The body does no blocking work; it only
    # creates a task.
    runner = get_sync_runner()
    if not runner.start():
        raise HTTPException(status_code=409, detail="A sync is already running")
    return {"status": "started"}


@router.get("/status", response_model=SyncStatusResponse)
def sync_status(ctx=Depends(get_read_context)) -> SyncStatusResponse:
    """Sync age and run state for the app shell's top bar.

    The ONLY metrics-cache age the UI shows; screens must not render their
    own. Read-only: touches the cache file and the runner flag, never Binance.
    """
    cache = MetricsCache(cache_path_for(ctx.config_manager))
    age = cache.age_seconds()
    auto = (ctx.config_manager.config.get("automation", {}) or {})
    sy = auto.get("auto_sync", {}) or {}
    try:
        minutes = int(sy.get("interval_minutes", 5))
    except (TypeError, ValueError):
        minutes = 5
    if minutes < 2:
        minutes = 2
    staleness = staleness_for(cache)
    staleness.is_stale = age is None or age > STALE_AFTER_INTERVALS * minutes * 60
    return SyncStatusResponse(
        is_running=get_sync_runner().is_running, staleness=staleness)


@router.get("/stream")
async def stream_sync() -> StreamingResponse:
    runner = get_sync_runner()

    async def event_source():
        async for event in runner.events():
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
