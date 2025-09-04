"""
Tests for CryptoPortfolioTracker initialization to ensure proper component setup.
"""

import pytest
from unittest.mock import Mock, patch

from crypto_portfolio_tracker.config import ConfigManager
from crypto_portfolio_tracker.portfolio_tracker import CryptoPortfolioTracker


def test_online_mode_initialization():
    """
    Test that CryptoPortfolioTracker can be instantiated in online mode without 
    dependency order issues.
    
    This test ensures that all components are properly initialized in the correct
    order, particularly that specialized components like data_synchronizer are
    available before methods that depend on them are called.
    """
    # Mock the config manager
    config_manager = Mock(spec=ConfigManager)
    config_manager.config = {
        "portfolio": {"testnet_mode": False},
        "apis": {"binance": {"timeout": 60}},
        "database": {"path": ":memory:", "cleanup_days": 90},
        "cache": {"path": "/tmp/test_cache"},
    }
    config_manager.get_database_path.return_value = ":memory:"
    config_manager.get_backup_dir.return_value = "/tmp/test_backups"
    config_manager.symbol_mapper = Mock()
    
    # Mock the database manager to avoid schema issues
    with patch("crypto_portfolio_tracker.portfolio_tracker.DatabaseManager") as mock_db_manager:
        mock_db_instance = Mock()
        mock_db_manager.return_value = mock_db_instance
        
        # Mock network-dependent operations to avoid actual network calls
        with patch("crypto_portfolio_tracker.portfolio_tracker.DataSynchronizer._init_binance_client") as mock_init_client:
            mock_init_client.return_value = Mock()  # Return a mock Binance client
            
            # This should not raise an AttributeError about missing data_synchronizer
            tracker = CryptoPortfolioTracker(config_manager, force_offline=False)
            
            # Verify that the tracker was created successfully
            assert tracker is not None
            assert hasattr(tracker, 'data_synchronizer')
            assert hasattr(tracker, 'portfolio_analyzer')
            assert hasattr(tracker, 'trade_executor')
            assert hasattr(tracker, 'dca_manager')
            assert hasattr(tracker, 'report_generator')
            assert hasattr(tracker, 'data_manager')


def test_offline_mode_initialization():
    """
    Test that CryptoPortfolioTracker can be instantiated in offline mode.
    """
    # Mock the config manager
    config_manager = Mock(spec=ConfigManager)
    config_manager.config = {
        "portfolio": {"testnet_mode": False},
        "apis": {"binance": {"timeout": 60}},
        "database": {"path": ":memory:", "cleanup_days": 90},
        "cache": {"path": "/tmp/test_cache"},
    }
    config_manager.get_database_path.return_value = ":memory:"
    config_manager.get_backup_dir.return_value = "/tmp/test_backups"
    config_manager.symbol_mapper = Mock()
    
    # Mock the database manager to avoid schema issues
    with patch("crypto_portfolio_tracker.portfolio_tracker.DatabaseManager") as mock_db_manager:
        mock_db_instance = Mock()
        mock_db_manager.return_value = mock_db_instance
        
        # This should not raise any errors
        tracker = CryptoPortfolioTracker(config_manager, force_offline=True)
        
        # Verify that the tracker was created successfully
        assert tracker is not None
        assert hasattr(tracker, 'data_synchronizer')
        assert hasattr(tracker, 'portfolio_analyzer')
        assert hasattr(tracker, 'trade_executor')
        assert hasattr(tracker, 'dca_manager')
        assert hasattr(tracker, 'report_generator')
        assert hasattr(tracker, 'data_manager')
        assert tracker.binance_client is None