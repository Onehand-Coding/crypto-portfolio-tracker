from typing import Optional

from pydantic import BaseModel, Field


class Staleness(BaseModel):
    """How old the served figures are. Never hidden from the UI."""

    cached_at: Optional[str] = Field(
        None, description="ISO timestamp of the last successful sync"
    )
    age_seconds: Optional[float] = Field(
        None, description="Seconds since that sync; null when never synced"
    )
    is_stale: bool = Field(
        description="True when older than the freshness threshold or never synced"
    )


class Environment(BaseModel):
    is_testnet: bool
    database_path: str
    label: str = Field(description="'TESTNET' or 'LIVE' -- rendered globally")


class SyncStatusResponse(BaseModel):
    """The single source of truth for sync age. Polled by the app shell's
    top bar; screens must not show their own metrics-cache age."""

    is_running: bool
    staleness: Staleness
