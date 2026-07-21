"""API-owned cache of the last successful portfolio metrics.

calculate_portfolio_metrics() reaches for live prices, so GET endpoints
read this file instead. It is written only by an explicit sync. Its age is
always exposed to the UI rather than hidden -- a stale figure the user can
see is safe; a stale figure presented as current is not.

This lives outside the core database. No core schema is modified.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

from api.serialization import jsonable

logger = logging.getLogger(__name__)


class MetricsCache:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, metrics: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {str(k): jsonable(v) for k, v in metrics.items()}
        payload["_cached_at"] = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self.path)

    def read(self) -> Optional[dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Metrics cache unreadable at %s: %s", self.path, exc)
            return None

    def age_seconds(self) -> Optional[float]:
        cached = self.read()
        if not cached or "_cached_at" not in cached:
            return None
        return time.time() - cached["_cached_at"]


def cache_path_for(config_manager) -> Path:
    """Testnet and live caches are separate files, mirroring the separate DBs.

    config.is_testnet_mode already switches the database path; the cache must
    switch with it or testnet figures would surface as live ones.
    """
    suffix = "testnet" if config_manager.is_testnet_mode else "live"
    return Path("data") / "api_cache" / f"metrics_{suffix}.json"


def analysis_cache_path(config_manager, kind: str) -> Path:
    """Cache for one kind of live analysis (rebalance, dca, profit, technical).

    These call the core's analysis methods, which need a live Binance client and
    fetch prices and klines -- far too slow and too failure-prone to run inside a
    page load. They follow the same contract as the metrics cache: an explicit
    user action computes and stores; reads are instant and always show the age.
    """
    suffix = "testnet" if config_manager.is_testnet_mode else "live"
    return Path("data") / "api_cache" / f"{kind}_{suffix}.json"
