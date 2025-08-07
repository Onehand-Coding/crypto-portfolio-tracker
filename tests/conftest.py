"""
Test configuration and fixtures for the crypto portfolio tracker.
"""

import pytest
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker
from crypto_portfolio_tracker.database import DatabaseManager


@pytest.fixture
def mock_config_manager():
    """Create a mock configuration manager with all required attributes."""
    config = Mock(spec=ConfigManager)

    # Mock all required attributes
    config.symbol_mapper = Mock()
    config.symbol_mapper.get_symbol_mapping.return_value = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
    }
    config.symbol_mapper.get_binance_symbol.return_value = "BTCUSDT"

    # Mock the config attribute (not get_config method)
    config.config = {
        "portfolio": {
            "testnet_mode": True,
            "live_trading_enabled": False,
            "target_allocation": {
                "BTC": 0.35,
                "ETH": 0.25,
                "SOL": 0.10,
                "RENDER": 0.10,
                "LINK": 0.10,
                "TAO": 0.10,
            },
        },
        "apis": {"binance": {"testnet": True, "timeout": 60}},
        "database": {"path": "test_portfolio.db", "backup_retention_days": 7},
    }

    # Mock other required methods
    config.get_database_path.return_value = Path("test_portfolio.db")
    config.get_backup_dir.return_value = Path("backups")
    config.is_live = False
    config.is_testnet_mode = True

    return config


@pytest.fixture
def mock_binance_client():
    """Create a mock Binance client."""
    client = Mock()

    # Mock account info
    client.get_account.return_value = {
        "balances": [
            {"asset": "BTC", "free": "0.1", "locked": "0.0"},
            {"asset": "ETH", "free": "1.0", "locked": "0.0"},
            {"asset": "USDT", "free": "1000.0", "locked": "0.0"},
        ]
    }

    # Mock trading info
    client.get_exchange_info.return_value = {
        "symbols": [
            {
                "symbol": "BTCUSDT",
                "status": "TRADING",
                "baseAsset": "BTC",
                "quoteAsset": "USDT",
                "filters": [
                    {
                        "filterType": "LOT_SIZE",
                        "minQty": "0.00001",
                        "maxQty": "1000",
                        "stepSize": "0.00001",
                    }
                ],
            }
        ]
    }

    return client


@pytest.fixture
def mock_symbol_mapper():
    """Create a mock symbol mapper."""
    mapper = Mock()
    mapper.get_symbol_mapping.return_value = {"BTC": "bitcoin", "ETH": "ethereum"}
    mapper.get_binance_symbol.return_value = "BTCUSDT"
    mapper.get_coingecko_id.return_value = "bitcoin"
    mapper.normalize_symbol.return_value = "BTC"

    return mapper


@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_portfolio.db"


@pytest.fixture
def db_manager(test_db_path):
    """Create a database manager for testing."""
    return DatabaseManager(
        db_path=test_db_path, backup_dir=test_db_path.parent / "backups", cleanup_days=7
    )


@pytest.fixture
def tracker(mock_config_manager):
    """Create a CryptoPortfolioTracker instance for testing."""
    # Mock the dependencies to avoid real API calls
    with patch(
        "crypto_portfolio_tracker.portfolio_tracker.BinanceFetcher"
    ) as mock_fetcher_class:
        mock_fetcher = Mock()
        mock_fetcher.fetch_binance_balances.return_value = pd.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "quantity": [0.1, 1.0],
                "value_usd": [5000, 3000],
            }
        )
        mock_fetcher.fetch_futures_balance.return_value = []
        mock_fetcher.fetch_funding_balance.return_value = []
        mock_fetcher.fetch_simple_earn_balances.return_value = {}
        mock_fetcher_class.return_value = mock_fetcher

        with patch(
            "crypto_portfolio_tracker.portfolio_tracker.PriceEnricher"
        ) as mock_enricher_class:
            mock_enricher = Mock()
            mock_enricher.get_current_prices = AsyncMock(
                return_value={"BTC": 50000, "ETH": 3000}
            )
            mock_enricher_class.return_value = mock_enricher

            with patch(
                "crypto_portfolio_tracker.portfolio_tracker.CryptoTrendAnalyzer"
            ) as mock_analyzer_class:
                mock_analyzer = Mock()
                mock_analyzer_class.return_value = mock_analyzer

                # Create the tracker instance with correct parameters
                tracker = CryptoPortfolioTracker(
                    config_manager=mock_config_manager,
                    force_offline=True,  # Force offline to avoid real API calls
                )

                # Mock the internal attributes
                tracker.fetcher = mock_fetcher
                tracker.enricher = mock_enricher
                tracker.analyzer = mock_analyzer

                return tracker


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return pd.DataFrame(
        {
            "symbol": ["BTC", "ETH", "SOL"],
            "quantity": [0.1, 1.0, 10.0],
            "value_usd": [5000, 3000, 1000],
            "cost_basis": [4500, 2800, 950],
            "allocation": [0.56, 0.33, 0.11],
        }
    )


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    return [
        {
            "symbol": "BTC",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "BUY",
            "quantity": 0.1,
            "price_usd": 50000.0,
            "fee_usd": 5.0,
            "transaction_hash": "hash1",
        },
        {
            "symbol": "ETH",
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "BUY",
            "quantity": 1.0,
            "price_usd": 3000.0,
            "fee_usd": 3.0,
            "transaction_hash": "hash2",
        },
    ]
