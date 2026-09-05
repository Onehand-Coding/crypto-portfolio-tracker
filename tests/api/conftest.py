"""Fixtures for API tests. The core is mocked; no network, no real DB."""

import os
from unittest.mock import Mock

import pytest

# Headless matplotlib: pinned before any api import pulls in pyplot (which
# reads MPLBACKEND at import time; a display backend aborts teardown here).
os.environ.setdefault("MPLBACKEND", "Agg")

# pyplot is usually ALREADY imported by now: the parent tests/conftest.py pulls
# in portfolio_tracker -> visualizations at collection, binding TkAgg before
# the setdefault above can take effect. So switch the live backend too -- once
# here at process start, never per request.
import matplotlib

if matplotlib.get_backend().lower() != "agg":
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")

from api import deps, sync_runner
from api import main as api_main


@pytest.fixture(autouse=True)
def _stub_auto_sync_scheduler(monkeypatch):
    """Never boot the real auto-sync scheduler in API tests.

    The app lifespan calls module-global build_scheduler(), which builds the
    real read context (real ConfigManager + real DatabaseManager on the dev
    data/*.db path) and spawns a real scheduler task. Any test using
    `with TestClient(app)` runs the lifespan, so without this stub those
    tests touch the dev database. The dummy's stop is awaitable because the
    lifespan awaits it. Tests that patch build_scheduler themselves (e.g.
    the lifespan test) override this stub via monkeypatch.
    """

    class _DummyScheduler:
        def start(self):
            return None

        async def stop(self):
            return None

    monkeypatch.setattr(api_main, "build_scheduler", lambda: _DummyScheduler())


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Every test starts with a clean singleton slate."""
    deps.reset_singletons()
    # The sync runner is a separate module-level singleton. Left over from a
    # previous test it reports is_running and the next POST /api/sync gets a
    # 409, so a route test would pass for the wrong reason.
    sync_runner._runner = None
    yield
    deps.reset_singletons()
    sync_runner._runner = None


@pytest.fixture
def mock_tracker():
    """A CryptoPortfolioTracker stand-in wired into the deps singleton."""
    tracker = Mock()
    tracker.config_manager = Mock()
    tracker.config_manager.is_testnet_mode = True
    deps.set_tracker_for_testing(tracker)
    return tracker


@pytest.fixture
def mock_read_context():
    """A ReadContext stand-in wired into the deps singleton."""
    context = Mock()
    deps.set_read_context_for_testing(context)
    return context
