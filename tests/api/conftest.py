"""Fixtures for API tests. The core is mocked; no network, no real DB."""

import pytest
from unittest.mock import Mock

from api import deps


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Every test starts with a clean singleton slate."""
    deps.reset_singletons()
    yield
    deps.reset_singletons()


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
