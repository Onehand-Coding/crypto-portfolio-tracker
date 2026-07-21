"""Fixtures for API tests. The core is mocked; no network, no real DB."""

from unittest.mock import Mock

import pytest

from api import deps, sync_runner


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
