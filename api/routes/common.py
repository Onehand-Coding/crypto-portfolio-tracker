"""Shared helpers for read routes."""

import datetime
from typing import Optional

from api.schemas.system import Staleness

STALE_AFTER_SECONDS = 3600.0


def staleness_for(cache) -> Staleness:
    """Staleness of a cache file, always reported rather than hidden."""
    age = cache.age_seconds()
    cached = cache.read() or {}
    cached_at = cached.get("_cached_at")
    return Staleness(
        cached_at=(datetime.datetime.fromtimestamp(cached_at).isoformat()
                   if cached_at else None),
        age_seconds=age,
        # No cache at all counts as stale: "never computed" must never render
        # as "current".
        is_stale=(age is None or age > STALE_AFTER_SECONDS),
    )


def num(value, default: float = 0.0) -> float:
    """Float or a default. NaN and None both mean 'not a number here'."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if result != result else result


def opt(value) -> Optional[float]:
    """Float or None. Never silently substitutes zero for unknown."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result
