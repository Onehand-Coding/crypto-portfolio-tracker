import pytest
import pandas as pd
import requests
import httpx


class DummyFetcher:
    def __init__(self):
        self.calls = []

    def fetch_binance_balances(self) -> pd.DataFrame:
        self.calls.append("spot")
        # Provide BTC and USDT balances
        return pd.DataFrame(
            [
                {"symbol": "BTC", "quantity": 0.01},
                {"symbol": "USDT", "quantity": 50.0},
            ]
        )

    def fetch_simple_earn_balances(self, *_args, **_kwargs):
        self.calls.append("earn")
        raise AssertionError("Earn balances should not be fetched in testnet mode")

    def fetch_futures_balance(self):
        self.calls.append("futures")
        raise AssertionError("Futures balance should not be fetched in testnet mode")

    def fetch_funding_balance(self):
        self.calls.append("funding")
        raise AssertionError("Funding balance should not be fetched in testnet mode")


class DummyBinanceClient:
    def get_asset_balance(self, asset: str):
        # Return a deterministic spot balance for USDT
        if asset == "USDT":
            return {"free": 100.0}
        return {"free": 0.0}


@pytest.mark.asyncio
async def test_testnet_skips_non_spot(monkeypatch):
    # Block any network accidentally
    def _requests_get(*_args, **_kwargs):
        raise AssertionError("Network call attempted via requests.get in testnet test")

    class _DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, *_args, **_kwargs):
            raise AssertionError("Network call attempted via httpx in testnet test")

    monkeypatch.setattr(requests, "get", _requests_get, raising=True)
    monkeypatch.setattr(httpx, "AsyncClient", _DummyAsyncClient, raising=True)

    from crypto_portfolio_tracker.config import ConfigManager
    from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker

    config_manager = ConfigManager()
    # Force testnet mode
    config_manager.config.setdefault("portfolio", {})["testnet_mode"] = True

    # Build tracker without network, then simulate online in testnet
    tracker = CryptoPortfolioTracker(config_manager, force_offline=True)
    tracker.offline_mode = False

    # Inject dummies
    dummy_fetcher = DummyFetcher()
    dummy_binance_client = DummyBinanceClient()
    tracker.fetcher = dummy_fetcher
    tracker.portfolio_analyzer.fetcher = dummy_fetcher
    tracker.binance_client = dummy_binance_client
    
    # Also set the config_manager and offline_mode on the portfolio analyzer
    tracker.portfolio_analyzer.config_manager = config_manager
    tracker.portfolio_analyzer.offline_mode = False
    
    # Set the binance client on the DCA manager
    tracker.dca_manager.binance_client = dummy_binance_client

    # Stub price enricher to avoid network and provide prices
    async def _stub_current_prices(symbols):
        # BTC at $50,000, USDT at $1
        return {s: (50000.0 if s == "BTC" else 1.0) for s in symbols}

    tracker.enricher.get_current_prices = _stub_current_prices  # type: ignore

    # Run metrics
    metrics = await tracker.calculate_portfolio_metrics()

    # Assert non-spot paths were not used
    assert "earn" not in tracker.fetcher.calls
    assert "futures" not in tracker.fetcher.calls
    assert "funding" not in tracker.fetcher.calls
    assert "spot" in tracker.fetcher.calls

    # Values should reflect spot only
    assert metrics["futures_value_usd"] == 0
    assert metrics["funding_value_usd"] == 0
    assert metrics["spot_earn_value_usd"] > 0  # from BTC and USDT

    # USDT availability should ignore funding/earn in testnet
    usdt = tracker.get_available_usdt_balance()
    assert usdt["spot_earn"] == 100.0  # from DummyBinanceClient
    assert usdt["funding"] == 0.0
    assert usdt["total"] == 100.0
