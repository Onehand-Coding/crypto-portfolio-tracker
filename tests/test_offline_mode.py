import pytest
import requests
import httpx


@pytest.mark.asyncio
async def test_offline_metrics_no_network(monkeypatch):
    """Ensure offline mode computes metrics without performing any network calls."""
    from crypto_portfolio_tracker.config import ConfigManager
    from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker

    # Any attempt to use requests/httpx should fail this test
    def _requests_get(*_args, **_kwargs):
        raise AssertionError("Network call attempted via requests.get in offline mode")

    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            raise AssertionError("Network call attempted via httpx in offline mode")

    monkeypatch.setattr(requests, "get", _requests_get, raising=True)
    monkeypatch.setattr(httpx, "AsyncClient", _DummyAsyncClient, raising=True)

    config_manager = ConfigManager()
    tracker = CryptoPortfolioTracker(config_manager, force_offline=True)

    # Sanity: in offline mode these should not be initialized
    assert tracker.binance_client is None
    assert tracker.fetcher is None

    metrics = await tracker.calculate_portfolio_metrics()

    # Ensure no network-derived sections are populated
    assert metrics["futures_value_usd"] == 0
    assert metrics["funding_value_usd"] == 0
    assert metrics["spot_earn_value_usd"] == 0
    assert metrics["futures_balances"] == []
    assert metrics["funding_balances"] == []

    # Holdings dataframe should exist (may be empty depending on DB state)
    assert "holdings_df" in metrics
