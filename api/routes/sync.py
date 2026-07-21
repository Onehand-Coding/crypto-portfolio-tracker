"""Explicit, user-initiated sync. The only path that contacts Binance."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.sync_runner import get_sync_runner

router = APIRouter(prefix="/api/sync", tags=["sync"])


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


@router.get("/stream")
async def stream_sync() -> StreamingResponse:
    runner = get_sync_runner()

    async def event_source():
        async for event in runner.events():
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_source(), media_type="text/event-stream")
